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

from .._base.errors import InsufficientCacheError, EpisodeDoneError
from ._base import BaseRewardTracer
from ._transition import TransitionBatch


__all__ = (
    'MonteCarlo',
)


class MonteCarlo(BaseRewardTracer):
    r"""
    A short-term cache for episodic Monte Carlo sampling.

    Parameters
    ----------
    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    """
    def __init__(self, gamma):
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._list = []
        self._done = False
        self._g = 0  # accumulator for return

    def add(self, s, a, r, done, logp=0.0, w=1.0):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly pop) before appending new transitions")

        self._list.append((s, a, r, logp, w))
        self._done = bool(done)
        if self._done:
            self._g = 0.  # init return

    def __len__(self):
        return len(self._list)

    def __bool__(self):
        return bool(len(self)) and self._done

    def pop(self):
        if not self:
            if not len(self):
                raise InsufficientCacheError(
                    "cache needs to receive more transitions before it can be popped from")
            else:
                raise InsufficientCacheError(
                    "cannot pop from cache before before receiving done=True")

        # pop state-action (propensities) pair
        s, a, r, logp, w = self._list.pop()

        # update return
        self._g = r + self.gamma * self._g

        return TransitionBatch.from_single(
            s=s, a=a, logp=logp, r=self._g, done=True, gamma=self.gamma,  # no bootstrapping
            s_next=s, a_next=a, logp_next=logp, w=w)                      # dummy values for *_next
