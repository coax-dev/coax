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
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from .._base.test_case import TestCase, DiscreteEnv, BoxEnv
from ..utils import safe_sample
from .q import Q


env_discrete = DiscreteEnv(random_seed=13)
env_boxspace = BoxEnv(random_seed=17)
num_quantiles = 3
quantile_embedding_dim = 4


def func_type3(S, tau, A, is_training):
    # TODO(frederik): implement
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1), jnp.ravel,
    ))
    S = hk.Flatten()(S)
    A = hk.Flatten()(A)
    X = jnp.concatenate((S, A), axis=-1)
    return seq(X)


def func_type4(S, quantiles, is_training):
    state_network = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training)
    ))
    x = state_network(S)
    state_vector_length = x.shape[-1]
    state_net_tiled = jnp.tile(x, [num_quantiles, 1])
    quantile_net = jnp.tile(quantiles, [1, quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = hk.Linear(state_vector_length)(quantile_net)
    quantile_net = jax.nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    x = hk.Linear(state_vector_length)(x)
    x = jax.nn.relu(x)
    x = hk.Linear(env_discrete.action_space.n)(x)
    return x


class TestQuantileQ(TestCase):
    decimal = 5

    def test_init(self):
        # cannot define a type-4 q-function on a non-discrete action space
        msg = r"type-4 q-functions are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            Q(func_type4, env_boxspace)

        # these should all be fine
        Q(func_type3, env_boxspace)
        Q(func_type3, env_discrete)
        Q(func_type4, env_discrete)
