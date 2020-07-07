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

import os

import numpy as onp
from gym.wrappers.frame_stack import LazyFrames

from .._base.mixins import (
    RandomStateMixin, SpaceUtilsMixin, SerializationMixin, LoggerMixin)
from .._base.errors import InsufficientCacheError
from .._core.transition import TransitionBatch
from ..utils import get_transition


__all__ = (
    'ExperienceReplayBuffer',
)


class ExperienceReplayBuffer(
        RandomStateMixin, SpaceUtilsMixin, SerializationMixin, LoggerMixin):
    r"""
    A simple numpy implementation of an experience replay buffer. This is
    written primarily with computer game environments (Atari) in mind.

    It implements a generic experience replay buffer for environments in which
    individual observations (frames) are stacked to represent the state.

    Parameters
    ----------
    env : gym environment

        The main gym environment. This is needed to extract some metadata such
        as shapes and dtypes.

    capacity : positive int

        The capacity of the experience replay buffer. DQN typically uses
        ``capacity=1000000``.

    n : positive int, optional

        The number of steps over which to perform bootstrapping, i.e.
        :math:`n`-step bootstrapping.

    gamma : float between 0 and 1

        Reward discount factor.

    random_seed : int or None

        To get reproducible results.


    """
    def __init__(
            self,
            env,
            capacity,
            n=1,
            gamma=0.99,
            random_seed=None):

        self.env = env
        self.capacity = int(capacity)
        self.n = int(n)
        self.gamma = float(gamma)
        self.random_seed = random_seed
        self._rnd = onp.random.RandomState(self.rng[0])
        self._ep_counter = 0

        # internal
        self._initialized = False

    def add(self, s, a, r, done, logp=0.0):
        r"""
        Add a transition to the experience replay buffer.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action

            A single action :math:`a`.

        r : float

            The observed rewards associated with this transition.

        done : bool

            Whether the episode has finished.

        logp : float, optional

            The log-propensity :math:`\log\pi(a|s)`.

        """
        if not self._initialized:
            self._init_cache(s, a, r)

        # extract last frame if 's' is a stack of frames
        if isinstance(s, LazyFrames):
            assert len(s) == self._num_stack, \
                "did gym.wrappers.FrameStack.num_stack change?"
            s = s[-1]

        # the use of index_update instead in-place updates is specific to JAX
        self._s[self._i] = s
        self._a[self._i] = a
        self._r[self._i] = r
        self._d[self._i] = done
        self._e[self._i] = self._ep_counter
        self._logp[self._i] = logp
        self._i = (self._i + 1) % (self.capacity + self.n)
        if self._num_transitions < self.capacity + self.n:
            self._num_transitions += 1
        if done:
            self._ep_counter += 1

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

            A :class:`TransitionBatch <coax.TransitionBatch>` object.

        """
        if not self._initialized or len(self) < batch_size:
            raise InsufficientCacheError(
                "insufficient cached data to sample from")

        S = []
        A = []
        logP = []
        Rn = []
        In = []
        S_next = []
        A_next = []
        logP_next = []

        for attempt in range(10 * batch_size):
            if not self._num_transitions:
                raise RuntimeError(
                    "please insert more transitions before sampling")
            j = int(self._rnd.randint(self._num_transitions))
            ep = self._e[j]                       # current episode id
            js = j - onp.arange(self._num_stack - 1, -1, -1)  # indices for s
            ks = js + self.n             # indices for s_next
            ls = j + onp.arange(self.n)  # indices for n-step rewards

            # wrap around
            js %= self.capacity + self.n
            ks %= self.capacity + self.n
            ls %= self.capacity + self.n

            # check if S indices are all from the same episode
            if onp.any(self._e[js[:-1]] > ep):
                # Check if all js are from the current episode or from the
                # immediately preceding episodes. Otherwise, we would generate
                # spurious data because it would probably mean that 'js' spans
                # the overwrite-boundary.
                continue
            for i in range(1, self._num_stack):
                # if j is from a previous episode, replace it by its successor
                if self._e[js[~i]] < ep:
                    js[~i] = js[~i + 1]
                if self._e[ks[~i]] < ep:
                    ks[~i] = ks[~i + 1]

            # gather partial returns
            rn = 0.
            gamma_t = 1.
            done = False
            for i in ls:
                rn += gamma_t * self._r[i]
                gamma_t *= self.gamma
                done = self._d[i]
                if done:
                    break
            rn = onp.expand_dims(rn, axis=-1)

            if not done and onp.any(self._e[ks] > ep):
                # this shouldn't happen (TODO: investigate)
                continue

            S.append(self._s[js])
            A.append(self._a[js[~0:]])
            logP.append(self._logp[js[~0:]])
            Rn.append(rn)
            S_next.append(self._s[ks])
            A_next.append(self._a[ks[~0:]])
            logP_next.append(self._logp[ks[~0:]])
            if done:
                In.append(onp.zeros(1))
            else:
                In.append(onp.power(
                    onp.expand_dims(self.gamma, axis=-1), self.n))

            if len(S) == batch_size:
                break

        if len(S) < batch_size:
            raise RuntimeError("couldn't construct valid sample")

        S = onp.stack(S, axis=0)
        A = onp.concatenate(A, axis=0)
        logP = onp.concatenate(logP, axis=0)
        Rn = onp.concatenate(Rn, axis=0)
        In = onp.concatenate(In, axis=0)
        S_next = onp.stack(S_next, axis=0)
        A_next = onp.concatenate(A_next, axis=0)
        logP_next = onp.concatenate(logP_next, axis=0)

        if self._num_stack == 1:
            S = onp.squeeze(S, axis=1)
            S_next = onp.squeeze(S_next, axis=1)

        return TransitionBatch(S, A, logP, Rn, In, S_next, A_next, logP_next)

    def clear(self):
        r"""
        Clear the experience replay buffer.

        """
        self._i = 0
        self._num_transitions = 0

    def __len__(self):
        return max(0, self._num_transitions - self.n)

    def __bool__(self):
        return bool(len(self))

    def save_state(self, filepath):
        assert isinstance(filepath, str), 'filepath must be a string'
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        # the state to store
        state_dict = {
            k: getattr(self, k) for k in (
                '_s', '_a', '_r', '_d', '_e', '_logp', '_i',
                '_num_transitions', 'capacity', 'n', 'gamma')}

        # make sure dir exists
        os.makedirs(os.path.abspath(os.path.dirname(filepath)), exist_ok=True)
        onp.savez(filepath, **state_dict)
        self.logger.info(f"saved experience-replay state to: {filepath}")

    def restore_state(self, filepath):
        try:
            if not self._initialized:
                tn = get_transition(self.env)
                self.add(tn.s, tn.a, tn.r, tn.done, 0)
            with onp.load(filepath) as f:
                for k, v in f.items():
                    setattr(self, k, v)
        except Exception:
            self.logger.error("failed to restore experience-replay state")
            raise

        self.logger.info(f"restored experience-replay state from: {filepath}")

    def _init_cache(self, s, a, r):
        # extract last frame if 's' is a stack of frames
        if isinstance(s, LazyFrames):
            self._num_stack = len(s)
            s = s[-1]
        else:
            self._num_stack = 1

        s = onp.asanyarray(s)
        a = onp.asanyarray(a)
        r = onp.asanyarray(r)
        self._i = 0
        self._num_transitions = 0

        # construct appropriate shapes
        n = (self.capacity + self.n,)

        # create cache attrs
        self._s = onp.empty(n + s.shape, s.dtype)
        self._a = onp.zeros(n + a.shape, a.dtype)
        self._r = onp.zeros(n + r.shape, r.dtype)
        self._d = onp.zeros(n, 'bool')
        self._logp = onp.zeros(n, 'float32')
        self._e = onp.zeros(n, 'int32')
        self._initialized = True
