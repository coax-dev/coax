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

from collections import deque
from itertools import islice

import numpy as onp

from .._base.errors import InsufficientCacheError, EpisodeDoneError
from ._base import BaseRewardTracer
from ._transition import TransitionBatch


__all__ = (
    'NStep',
)


class NStep(BaseRewardTracer):
    r"""
    A short-term cache for :math:`n`-step bootstrapping.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.

    record_extra_info : bool, optional

        Store all states, actions and rewards in the `extra_info` field
        of the `TransitionBatch`, e.g. for :code:`coax.regularizers.NStepEntropyRegularizer`.
    """

    def __init__(self, n, gamma, record_extra_info=False):
        self.n = int(n)
        self.gamma = float(gamma)
        self.record_extra_info = record_extra_info
        self.reset()

    def reset(self):
        self._deque_s = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = onp.power(self.gamma, onp.arange(self.n))
        self._gamman = onp.power(self.gamma, self.n)

    def add(self, s, a, r, done, logp=0.0, w=1.0):
        if self._done and len(self):
            raise EpisodeDoneError(
                "please flush cache (or repeatedly call popleft) before appending new transitions")

        self._deque_s.append((s, a, logp, w))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_s)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        if not self:
            raise InsufficientCacheError(
                "cache needs to receive more transitions before it can be popped from")

        # pop state-action (propensities) pair
        s, a, logp, w = self._deque_s.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        r = self._deque_r.popleft()

        # keep in mind that we've already popped (s, a, logp)
        if len(self) >= self.n:
            s_next, a_next, logp_next, _ = self._deque_s[self.n - 1]
            done = False
        else:
            # no more bootstrapping
            s_next, a_next, logp_next, done = s, a, logp, True

        extra_info = self._pop_extra_info(
            s, a, r, done, logp, w) if self.record_extra_info else None

        return TransitionBatch.from_single(
            s=s, a=a, logp=logp, r=rn, done=done, gamma=self._gamman,
            s_next=s_next, a_next=a_next, logp_next=logp_next, w=w, extra_info=extra_info)

    def _pop_extra_info(self, s, a, r, done, logp, w):
        last_s = s
        last_a = a
        last_r = r
        last_done = done
        last_logp = logp
        last_w = w
        states = []
        actions = []
        rewards = []
        dones = []
        log_props = []
        weights = []
        for i in range(self.n + 1):
            states.append(last_s)
            actions.append(last_a)
            rewards.append(last_r)
            dones.append(last_done)
            log_props.append(last_logp)
            weights.append(last_w)
            if not done and i < len(self._deque_s):
                last_s, last_a, last_logp, last_w = self._deque_s[i]
                last_r = self._deque_r[i]
                last_done = False
            else:
                last_done = True
        assert len(states) == len(actions) == len(
            rewards) == len(dones) == len(log_props) == len(weights) == self.n + 1
        extra_info = {'states': states, 'actions': actions,
                      'rewards': rewards, 'dones': dones,
                      'log_props': log_props, 'weights': weights}
        return {k: tuple(v) for k, v in extra_info.items()}
