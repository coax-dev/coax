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

from abc import ABC, abstractmethod
from collections import deque
from itertools import islice

import jax
import numpy as onp

from .._base.errors import InsufficientCacheError, EpisodeDoneError
from .._base.mixins import SpaceUtilsMixin

from .._core.transition import TransitionSingle


__all__ = (
    'MonteCarloCache',
    'NStepCache',
)


class BaseShortTermCache(ABC, SpaceUtilsMixin):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def reset(self):
        r"""
        Reset the cache to the initial state.

        """
        pass

    @abstractmethod
    def add(self, s, a, r, done, logp=0.0):
        r"""
        Add a transition to the experience cache.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        logp : float, optional

            The log-propensity :math:`\log\pi(a|s)`.

        """
        pass

    @abstractmethod
    def pop(self):
        r"""
        Pop a single transition from the cache.

        Returns
        -------
        transition : TransitionBatch

            A :class:`TransitionBatch <coax.TransitionBatch>` object with
            ``batch_size=1``.

        """
        pass

    def flush(self):
        r"""
        Flush all transitions from the cache.

        Returns
        -------
        transitions : TransitionBatch

            A :class:`TransitionBatch <coax.TransitionBatch>` object.

        """
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be "
                "flushed")

        transitions = []

        while self:
            transitions.append(self.pop())

        return jax.tree_multimap(
            lambda *leaves: onp.concatenate(leaves, axis=0), *transitions)


class NStepCache(BaseShortTermCache):
    r"""
    A short-term cache for :math:`n`-step bootstrapping.

    Parameters
    ----------
    env : gym environment

        The main gym environment. This is needed to determine ``num_actions``.

    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    """
    def __init__(self, env, n, gamma):
        super().__init__(env)
        self.n = int(n)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._deque_sap = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = onp.power(self.gamma, onp.arange(self.n))
        self._gamman = onp.power(self.gamma, self.n)

    def add(self, s, a, r, done, logp=0.0):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly call popleft) before "
                "appending new transitions")

        self._deque_sap.append((s, a, logp))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_sap)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be "
                "popped from")

        # pop state-action (propensities) pair
        s, a, logp = self._deque_sap.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        self._deque_r.popleft()

        # keep in mind that we've already popped (s, a, logp)
        if len(self) >= self.n:
            s_next, a_next, logp_next = self._deque_sap[self.n - 1]
            done = False
        else:
            # no more bootstrapping
            s_next, a_next, logp_next, done = s, a, logp, True

        transition = TransitionSingle(
            s=s, a=a, logp=logp, r=rn, done=done,
            s_next=s_next, a_next=a_next, logp_next=logp_next)

        return transition.to_batch(gamma=self._gamman)


class MonteCarloCache(BaseShortTermCache):
    r"""
    A short-term cache for episodic Monte Carlo sampling.

    Parameters
    ----------
    env : gym environment

        The main gym environment. This is needed to determine ``num_actions``.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    """
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._list = []
        self._done = False
        self._g = 0  # accumulator for return

    def add(self, s, a, r, done, logp=0.0):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly pop) before appending new "
                "transitions")

        self._list.append((s, a, r, logp))
        self._done = bool(done)
        if done:
            self._g = 0.  # init return

    def __len__(self):
        return len(self._list)

    def __bool__(self):
        return bool(len(self)) and self._done

    def pop(self):
        if not self:
            if not len(self):
                raise InsufficientCacheError(
                    "cache needs to receive more transitions before it can be "
                    "popped from")
            else:
                raise InsufficientCacheError(
                    "cannot pop from cache before before receiving done=True")

        # pop state-action (propensities) pair
        s, a, r, logp = self._list.pop()

        # update return
        self._g = r + self.gamma * self._g

        transition = TransitionSingle(
            s=s, a=a, logp=logp, r=self._g, done=True,  # no bootstrapping
            s_next=s, a_next=a, logp_next=logp)         # dummy values

        return transition.to_batch(gamma=self.gamma)
