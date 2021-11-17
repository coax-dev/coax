from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp

from .._base.mixins import CopyMixin
from ..utils import pretty_repr


__all__ = (
    'TransitionBatch',
)


class TransitionBatch(CopyMixin):
    r"""

    A container object for a batch of MDP transitions.

    Parameters
    ----------

    S : pytree with ndarray leaves

        A batch of state observations :math:`S_t`.

    A : ndarray

        A batch of actions :math:`A_t`.

    logP : ndarray

        A batch of log-propensities :math:`\log\pi(A_t|S_t)`.

    Rn : ndarray

        A batch of partial (:math:`\gamma`-discounted) returns. For instance,
        in :math:`n`-step bootstrapping these are given by:

        .. math::

            R^{(n)}_t\ &=\ \sum_{k=0}^{n-1}\gamma^kR_{t+k} \\

        In other words, it's the part of the :math:`n`-step return *without*
        the bootstrapping term.

    In : ndarray

        A batch of bootstrap factors. For instance, in :math:`n`-step
        bootstrapping these are given by :math:`I^{(n)}_t=\gamma^n` when
        bootstrapping and :math:`I^{(n)}_t=0` otherwise. Bootstrap factors are
        used in constructing the :math:`n`-step bootstrapped target:

        .. math::

            G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,Q(S_{t+1}, A_{t+1})

    S_next : pytree with ndarray leaves

        A batch of next-state observations :math:`S_{t+n}`. This is typically
        used to contruct the TD target in :math:`n`-step bootstrapping.

    A_next : ndarray, optional

        A batch of next-actions :math:`A_{t+n}`. This is typically used to
        contruct the TD target in :math:`n`-step bootstrapping when using SARSA
        updates.

    logP_next : ndarray, optional

        A batch of log-propensities :math:`\log\pi(A_{t+n}|S_{t+n})`.

    W : ndarray, optional

        A batch of importance weights associated with the sampling procedure that generated each
        transition. For example, we need these values when we sample transitions from a
        :class:`PrioritizedReplayBuffer <coax.experience_replay.PrioritizedReplayBuffer>`.

    """
    __slots__ = ('S', 'A', 'logP', 'Rn', 'In', 'S_next',
                 'A_next', 'logP_next', 'W', 'idx', 'extra_info')

    def __init__(self, S, A, logP, Rn, In, S_next, A_next=None, logP_next=None, W=None, idx=None,
                 extra_info=None):

        self.S = S
        self.A = A
        self.logP = logP
        self.Rn = Rn
        self.In = In
        self.S_next = S_next
        self.A_next = A_next
        self.logP_next = logP_next
        self.W = onp.ones_like(Rn) if W is None else W
        self.idx = onp.arange(Rn.shape[0], dtype='int32') if idx is None else idx
        self.extra_info = extra_info

    @classmethod
    def from_single(
            cls, s, a, logp, r, done, gamma,
            s_next=None, a_next=None, logp_next=None, w=1, idx=None, extra_info=None):
        r"""

        Create a TransitionBatch (with batch_size=1) from a single transition.

        Attributes
        ----------
        s : state observation

            A single state observation :math:`S_t`.

        a : action

            A single action :math:`A_t`.

        logp : non-positive float

            The log-propensity :math:`\log\pi(A_t|S_t)`.

        r : float or array of floats

            A single reward :math:`R_t`.

        done : bool

            Whether the episode has finished.

        info : dict or None

            Some additional info about the current time step.

        s_next : state observation

            A single next-state observation :math:`S_{t+1}`.

        a_next : action

            A single next-action :math:`A_{t+1}`.

        logp_next : non-positive float

            The log-propensity :math:`\log\pi(A_{t+1}|S_{t+1})`.

        w : positive float, optional

            The importance weight associated with the sampling procedure that generated this
            transition.

        idx : int, optional

            The identifier of this particular transition.

        """

        # check types
        array = (int, float, onp.ndarray, jnp.ndarray)
        if not (isinstance(logp, array) and onp.all(logp <= 0)):
            raise TypeError(f"logp must be non-positive float(s), got: {logp}")
        if not isinstance(r, array):
            raise TypeError(f"r must be a scalar or an array, got: {r}")
        if not isinstance(done, bool):
            raise TypeError(f"done must be a bool, got: {done}")
        if not (isinstance(gamma, (float, int)) and 0 <= gamma <= 1):
            raise TypeError(f"gamma must be a float in the unit interval [0, 1], got: {gamma}")
        if not (logp_next is None or (isinstance(logp_next, array) and onp.all(logp_next <= 0))):
            raise TypeError(f"logp_next must be None or non-positive float(s), got: {logp_next}")
        if not (isinstance(w, (float, int)) and w > 0):
            raise TypeError(f"w must be a positive float, got: {w}")

        return cls(
            S=_single_to_batch(s),
            A=_single_to_batch(a),
            logP=_single_to_batch(logp),
            Rn=_single_to_batch(r),
            In=_single_to_batch(float(gamma) * (1. - bool(done))),
            S_next=_single_to_batch(s_next) if s_next is not None else None,
            A_next=_single_to_batch(a_next) if a_next is not None else None,
            logP_next=_single_to_batch(logp_next) if logp_next is not None else None,
            W=_single_to_batch(float(w)),
            idx=_single_to_batch(idx) if idx is not None else None,
            extra_info=_single_to_batch(extra_info) if extra_info is not None else None
        )

    @property
    def batch_size(self):
        return onp.shape(self.Rn)[0]

    def to_singles(self):
        r"""

        Get an iterator of single transitions.

        Returns
        -------
        transition_batches : iterator of TransitionBatch

            An iterator of :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` objects
            with ``batch_size=1``.

            **Note:** The iterator walks through the individual transitions *in reverse order*.

        """
        if self.batch_size == 1:
            yield self
            return  # break out of generator

        def lookup(i, pytree):
            s = slice(i, i + 1)  # ndim-preserving lookup
            return jax.tree_map(lambda leaf: leaf[s], pytree)

        for i in range(self.batch_size):
            yield TransitionBatch(*map(partial(lookup, i), self))

    def items(self):
        for k in self.__slots__:
            yield k, getattr(self, k)

    def _asdict(self):
        return dict(self.items())

    def __repr__(self):
        return pretty_repr(self)

    def __iter__(self):
        return (getattr(self, a) for a in self.__slots__)

    def __getitem__(self, int_or_slice):
        return tuple(self).__getitem__(int_or_slice)

    def __eq__(self, other):
        return (type(self) is type(other)) and all(
            onp.allclose(a, b) if isinstance(a, (onp.ndarray, jnp.ndarray))
            else (a is b if a is None else a == b)
            for a, b in zip(jax.tree_leaves(self), jax.tree_leaves(other)))


def _single_to_batch(pytree):
    # notice that we're pulling eveyrthing out of jax.numpy and into ordinary numpy land
    return jax.tree_map(lambda arr: onp.expand_dims(arr, axis=0), pytree)


jax.tree_util.register_pytree_node(
    TransitionBatch,
    lambda tn: (tuple(tn), None),
    lambda treedef, leaves: TransitionBatch(*leaves))
