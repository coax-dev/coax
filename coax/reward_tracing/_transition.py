# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

from functools import partial

import jax
import numpy as onp
import pandas as pd


__all__ = (
    'TransitionSingle',
    'TransitionBatch',
)


class BaseTransition:
    def to_series(self):
        d = {k: getattr(self, k) for k in self.__slots__}
        d = {k: v for k, v in d.items() if v is not None}
        s = pd.Series(d, name='', dtype='O')
        s.index.name = self.__class__.__name__
        return s

    def to_frame(self):
        s = self.to_series()
        df = s.rename('value').to_frame()
        df['shape'] = s.map(lambda x: jax.tree_map(
            lambda y: getattr(y, 'shape'), x))
        df['dtype'] = s.map(lambda x: jax.tree_map(
            lambda y: getattr(y, 'dtype'), x))
        return df

    def __repr__(self):
        with pd.option_context("display.max_columns", None), \
                pd.option_context("display.max_colwidth", 36):
            return repr(self.to_frame())

    def _repr_html_(self):
        with pd.option_context("display.max_columns", None), \
                pd.option_context("display.max_colwidth", 36):
            return self.to_frame()._repr_html_()

    def __iter__(self):
        return (getattr(self, a) for a in self.__slots__)

    def __getitem__(self, int_or_slice):
        return tuple(self).__getitem__(int_or_slice)


class TransitionSingle(BaseTransition):
    r"""

    A container object for a single transition in the MDP.

    Attributes
    ----------
    s : state observation

        A single state observation :math:`S_t`.

    a : action

        A single action :math:`A_t`.

    logp : ndarray

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

    logp_next : ndarray

        The log-propensity :math:`\log\pi(A_{t+1}|S_{t+1})`.

    """
    __slots__ = ('s', 'a', 'logp', 'r', 'done', 'info', 's_next', 'a_next', 'logp_next')

    def __init__(self, s, a, logp, r, done, info=None, s_next=None, a_next=None, logp_next=None):
        self.s = s
        self.a = a
        self.logp = logp
        self.r = r
        self.done = done
        self.info = info
        self.s_next = s_next
        self.a_next = a_next
        self.logp_next = logp_next

    @staticmethod
    def _to_batch(pytree):
        if pytree is None:
            return None
        return jax.tree_map(lambda x: onp.expand_dims(x, axis=0), pytree)

    def to_batch(self, gamma=0.9):
        s, a, logp, r, done, info, s_next, a_next, logp_next = self
        return TransitionBatch(
            S=self._to_batch(s),
            A=self._to_batch(a),
            logP=self._to_batch(logp),
            Rn=self._to_batch(r),
            In=self._to_batch(gamma * (1 - done)),
            S_next=self._to_batch(s_next),
            A_next=self._to_batch(a_next),
            logP_next=self._to_batch(logp_next),
        )


class TransitionBatch(BaseTransition):
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

    """
    __slots__ = ('S', 'A', 'logP', 'Rn', 'In', 'S_next', 'A_next', 'logP_next')

    def __init__(self, S, A, logP, Rn, In, S_next, A_next=None, logP_next=None):
        self.S = S
        self.A = A
        self.logP = logP
        self.Rn = Rn
        self.In = In
        self.S_next = S_next
        self.A_next = A_next
        self.logP_next = logP_next

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
        def lookup(i, pytree):
            s = slice(i, i + 1)  # ndim-preserving lookup
            return jax.tree_map(lambda leaf: leaf[s], pytree)

        for i in reversed(range(self.batch_size)):
            yield TransitionBatch(*map(partial(lookup, i), self))


jax.tree_util.register_pytree_node(
    TransitionSingle,
    lambda tn: (tuple(tn), None),
    lambda treedef, leaves: TransitionSingle(*leaves))


jax.tree_util.register_pytree_node(
    TransitionBatch,
    lambda tn: (tuple(tn), None),
    lambda treedef, leaves: TransitionBatch(*leaves))
