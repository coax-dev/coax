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
import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix

from ..utils import argmax
from .func_approx_base import BaseFuncApprox


__all__ = (
    'FuncApprox',
)


class FuncApprox(BaseFuncApprox):
    __doc__ = BaseFuncApprox.__doc__

    def __init__(
            self, env,
            random_seed=None,
            example_observation=None,
            example_action=None,
            **optimizer_kwargs):

        super().__init__(env, random_seed, **optimizer_kwargs)
        self._init_state(example_observation, example_action)

    def body(self, S, is_training=False):
        discrete = isinstance(self.env.observation_space, gym.spaces.Discrete)
        if discrete and S.ndim == 1 and jnp.issubdtype(S.dtype, jnp.integer):
            S = hk.one_hot(S, self.env.observation_space.n)
        return S

    def head_v(self, X_s, is_training=False):
        return hk.Linear(1, w_init=jnp.zeros)(X_s)

    def head_q1(self, X_sa, is_training=False):
        return hk.Linear(1, w_init=jnp.zeros)(X_sa)

    def head_q2(self, X_s, is_training=False):
        if self.action_space_is_discrete:
            return hk.Linear(self.num_actions, w_init=jnp.zeros)(X_s)

        raise NotImplementedError(
            "The default implementation of FuncApprox.head_q2 can only handle "
            "Discrete action spaces")

    def head_pi(self, X_s, is_training=False):
        if self.action_space_is_discrete:
            logits = hk.Linear(
                self.num_actions, w_init=jnp.zeros)
            return {'logits': logits(X_s)}

        if self.action_space_is_box:
            mu = hk.Linear(self.action_shape_flat, w_init=jnp.zeros)
            logvar = hk.Linear(self.action_shape_flat, w_init=jnp.zeros)
            return {'mu': mu(X_s), 'logvar': logvar(X_s)}

        raise NotImplementedError(
            "The default implementation of FuncApprox.head_pi "
            "can only handle Discrete and Box action spaces")

    def state_action_combiner(self, X_s, X_a, is_training=False):
        X_sa = jax.vmap(jnp.kron)(X_s, X_a)
        return hk.Flatten()(X_sa)

    def action_preprocessor(self, A):
        X_a = A  # default: no-op

        if self.action_space_is_discrete:
            isidx = A.ndim == 1 and jnp.issubdtype(A.dtype, jnp.integer)
            if isidx:
                X_a = hk.one_hot(A, self.num_actions)
            else:
                # check if A is indeed already one-hot encoded
                assert A.ndim == 2, A
                assert A.shape[1] == self.num_actions, A

        return X_a

    def action_postprocessor(self, X_a):
        A = X_a  # default: no-op

        if self.action_space_is_discrete:
            isidx = X_a.ndim == 1 and jnp.issubdtype(X_a.dtype, jnp.integer)
            if not isidx:
                # check if X_a is indeed one-hot encoded
                assert X_a.ndim == 2
                assert X_a.shape[1] == self.num_actions
                A = argmax(hk.next_rng_key(), X_a, axis=-1)

        return A

    def optimizer(self):
        return optix.sgd(**self.optimizer_kwargs)
