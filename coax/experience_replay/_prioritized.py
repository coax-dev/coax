import jax
import numpy as onp
import chex

from ..reward_tracing import TransitionBatch
from ..utils import SumTree
from ._base import BaseReplayBuffer


__all__ = (
    'PrioritizedReplayBuffer',
)


class PrioritizedReplayBuffer(BaseReplayBuffer):
    r"""

    A simple ring buffer for experience replay, with prioritized sampling.

    This class uses *proportional* sampling, which means that the transitions are sampled with
    relative probability :math:`p_i` defined as:

    .. math::

        p_i\ =\ \frac
            {\left(|\mathcal{A}_i| + \epsilon\right)^\alpha}
            {\sum_{j=1}^N \left(|\mathcal{A}_j| + \epsilon\right)^\alpha}

    Here :math:`\mathcal{A}_i` are advantages provided at insertion time and :math:`N` is the
    capacity of the buffer, which may be quite large. The :math:`\mathcal{A}_i` are typically just
    TD errors collected from a value-function updater, e.g. :func:`QLearning.td_error
    <coax.td_learning.QLearning.td_error>`.

    Since the prioritized samples are biased, the :attr:`sample` method also produces non-trivial
    importance weights (stored in the :class:`TransitionBatch.W
    <coax.reward_tracing.TransitionBatch>` attribute). The logic for constructing these weights for
    a sample of batch size :math:`n` is:

    .. math::

        w_i\ =\ \frac{\left(Np_i\right)^{-\beta}}{\max_{j=1}^n \left(Np_j\right)^{-\beta}}

    See section 3.4 of https://arxiv.org/abs/1511.05952 for more details.

    Parameters
    ----------
    capacity : positive int

        The capacity of the experience replay buffer.

    alpha : positive float, optional

        The sampling temperature :math:`\alpha>0`.

    beta : positive float, optional

        The importance-weight exponent :math:`\beta>0`.

    epsilon : positive float, optional

        The small regulator :math:`\epsilon>0`.

    random_seed : int, optional

        To get reproducible results.

    """
    def __init__(self, capacity, alpha=1.0, beta=1.0, epsilon=1e-4, random_seed=None):
        if not (isinstance(capacity, int) and capacity > 0):
            raise TypeError(f"capacity must be a positive int, got: {capacity}")
        if not (isinstance(alpha, (float, int)) and alpha > 0):
            raise TypeError(f"alpha must be a positive float, got: {alpha}")
        if not (isinstance(beta, (float, int)) and beta > 0):
            raise TypeError(f"beta must be a positive float, got: {beta}")
        if not (isinstance(epsilon, (float, int)) and epsilon > 0):
            raise TypeError(f"epsilon must be a positive float, got: {epsilon}")

        self._capacity = int(capacity)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._epsilon = float(epsilon)
        self._random_seed = random_seed
        self._rnd = onp.random.RandomState(random_seed)
        self.clear()  # sets: self._deque, self._index

    @property
    def capacity(self):
        return self._capacity

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        if not (isinstance(new_alpha, (float, int)) and new_alpha > 0):
            raise TypeError(f"alpha must be a positive float, got: {new_alpha}")
        if onp.isclose(new_alpha, self._alpha, rtol=0.01):
            return  # noop if new value is too close to old value (not worth the computation cost)
        new_values = onp.where(
            self._sumtree.values <= 0, 0.,  # only change exponents for positive values
            onp.exp(onp.log(onp.maximum(self._sumtree.values, 1e-15)) * (new_alpha / self._alpha)))
        self._sumtree.set_values(..., new_values)
        self._alpha = float(new_alpha)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, new_beta):
        if not (isinstance(new_beta, (float, int)) and new_beta > 0):
            raise TypeError(f"beta must be a positive float, got: {new_beta}")
        self._beta = float(new_beta)

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        if not (isinstance(new_epsilon, (float, int)) and new_epsilon > 0):
            raise TypeError(f"epsilon must be a positive float, got: {new_epsilon}")
        self._epsilon = float(new_epsilon)

    def add(self, transition_batch, Adv):
        r"""

        Add a transition to the experience replay buffer.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

        Adv : ndarray

            A batch of advantages, used to construct the priorities :math:`p_i`.

        """
        if not isinstance(transition_batch, TransitionBatch):
            raise TypeError(
                f"transition_batch must be a TransitionBatch, got: {type(transition_batch)}")

        transition_batch.idx = self._index + onp.arange(transition_batch.batch_size)
        idx = transition_batch.idx % self.capacity  # wrap around
        chex.assert_equal_shape([idx, Adv])
        self._storage[idx] = list(transition_batch.to_singles())
        self._sumtree.set_values(idx, onp.power(onp.abs(Adv) + self.epsilon, self.alpha))
        self._index += transition_batch.batch_size

    def sample(self, batch_size=32):
        r"""

        Get a batch of transitions to be used for bootstrapped updates.

        Parameters
        ----------
        batch_size : positive int, optional

            The desired batch size of the sample.

        Returns
        -------
        transitions : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

        """
        idx = self._sumtree.sample(n=batch_size)
        P = self._sumtree.values[idx] / self._sumtree.root_value  # prioritized, biased propensities
        W = onp.power(P * len(self), -self.beta)                  # inverse propensity weights (β≈1)
        W /= W.max()  # for stability, ensure only down-weighting (see sec. 3.4 of arxiv:1511.05952)
        transition_batch = _concatenate_leaves(self._storage[idx])
        chex.assert_equal_shape([transition_batch.W, W])
        transition_batch.W *= W
        return transition_batch

    def update(self, idx, Adv):
        r"""

        Update the priority weights of transitions previously added to the buffer.

        Parameters
        ----------
        idx : 1d array of ints

            The identifiers of the transitions to be updated.

        Adv : ndarray

            The corresponding updated advantages.

        """
        idx = onp.asarray(idx, dtype='int32')
        Adv = onp.asarray(Adv, dtype='float32')
        chex.assert_equal_shape([idx, Adv])
        chex.assert_rank([idx, Adv], 1)

        idx_lookup = idx % self.capacity  # wrap around
        new_values = onp.where(
            _get_transition_batch_idx(self._storage[idx_lookup]) == idx,  # only update if ids match
            onp.power(onp.abs(Adv) + self.epsilon, self.alpha),
            self._sumtree.values[idx_lookup])
        self._sumtree.set_values(idx_lookup, new_values)

    def clear(self):
        r""" Clear the experience replay buffer. """
        self._storage = onp.full(shape=(self.capacity,), fill_value=None, dtype='object')
        self._sumtree = SumTree(capacity=self.capacity)
        self._index = 0

    def __len__(self):
        return min(self.capacity, self._index)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._storage[:len(self)])


def _concatenate_leaves(pytrees):
    return jax.tree_multimap(lambda *leaves: onp.concatenate(leaves, axis=0), *pytrees)


@onp.vectorize
def _get_transition_batch_idx(transition):
    return transition.idx
