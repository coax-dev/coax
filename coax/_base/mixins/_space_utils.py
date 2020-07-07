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

import gym
import numpy as onp
import jax
import jax.numpy as jnp
from gym.wrappers.frame_stack import LazyFrames


class SpaceUtilsMixin:
    r""" this mixin class holds all space-dependent utils """
    @property
    def action_space_is_box(self):
        return isinstance(self.env.action_space, gym.spaces.Box)

    @property
    def action_space_is_discrete(self):
        return isinstance(self.env.action_space, gym.spaces.Discrete)

    @property
    def action_shape(self):
        assert hasattr(self, 'env') and hasattr(self.env, 'action_space')
        if not hasattr(self, '_action_shape'):
            action = self.env.action_space.sample()
            self._action_shape = jax.tree_map(jnp.shape, action)
        return self._action_shape

    @property
    def action_shape_flat(self):
        if not hasattr(self, '_action_shape_flat'):
            self._action_shape_flat = int(onp.prod(self.action_shape))
        return self._action_shape_flat

    @property
    def num_actions(self):
        if not hasattr(self, '_num_actions'):
            if not self.action_space_is_discrete:
                raise AttributeError(
                    "num_actions attribute is inaccessible; does the env "
                    "have a Discrete action space?")
            self._num_actions = self.env.action_space.n
        return self._num_actions

    def _postprocess_action(self, a):
        if self.action_space_is_discrete:
            return int(a)
        if self.action_space_is_box:
            lo = self.env.action_space.low
            hi = self.env.action_space.high
            return onp.clip(a, lo, hi)
        return a

    @staticmethod
    def _preprocess_state(s):
        if isinstance(s, LazyFrames):
            return onp.asanyarray(s)
        return s
