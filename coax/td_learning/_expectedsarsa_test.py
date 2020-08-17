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

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
from jax.experimental.optix import sgd

from .._base.test_case import TestCase, DiscreteEnv
from .._core.value_q import Q
from .._core.value_based_policy import BoltzmannPolicy
from ..utils import get_transition
from ._expectedsarsa import ExpectedSarsa

env = DiscreteEnv(random_seed=13)


def func_type1(S, A, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1), jnp.ravel,
    ))
    flatten = hk.Flatten()
    X = jnp.concatenate((flatten(S), flatten(A)), axis=-1)
    return seq(X)


def func_type2(S, is_training):
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n),
    ))
    return seq(S)


class TestExpectedSarsa(TestCase):

    def setUp(self):

        self.transition_batch = get_transition(self.env_discrete).to_batch()

    def test_update_type1_discrete(self):
        q = Q(func_type1, env.observation_space, env.action_space)
        q_targ = q.copy()
        pi_targ = BoltzmannPolicy(q_targ)
        updater = ExpectedSarsa(q, pi_targ, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_batch)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_type2_discrete(self):
        q = Q(func_type2, env.observation_space, env.action_space)
        q_targ = q.copy()
        pi_targ = BoltzmannPolicy(q_targ)
        updater = ExpectedSarsa(q, pi_targ, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_batch)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)
