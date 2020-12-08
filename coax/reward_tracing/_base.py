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

import jax
import numpy as onp

from .._base.errors import InsufficientCacheError


__all__ = (
    'BaseRewardTracer',
)


class BaseRewardTracer(ABC):

    @abstractmethod
    def reset(self):
        r"""
        Reset the cache to the initial state.

        """
        pass

    @abstractmethod
    def add(self, s, a, r, done, logp=0.0, w=1.0):
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

        w : float, optional

            Sample weight associated with the given state-action pair.

        """
        pass

    @abstractmethod
    def pop(self):
        r"""
        Pop a single transition from the cache.

        Returns
        -------
        transition : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object with
            ``batch_size=1``.

        """
        pass

    def flush(self):
        r"""
        Flush all transitions from the cache.

        Returns
        -------
        transitions : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

        """
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be flushed")

        transitions = []

        while self:
            transitions.append(self.pop())

        return jax.tree_multimap(lambda *leaves: onp.concatenate(leaves, axis=0), *transitions)
