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

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from .._base.test_case import TestCase
from ..utils import get_transition
from .func_approx import FuncApprox


def primes(xmin=1):
    x = xmin
    while True:
        if onp.all(x % onp.arange(2, x // 2)):
            yield x
        x += 1


class MyFuncApprox(FuncApprox):
    def body(self, S, is_training):
        X = hk.Linear(7)(S)
        X = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(X, is_training)
        X = jax.nn.relu(X)
        return X


class TestFuncApprox(TestCase):
    def setUp(self):
        self.rngs = hk.PRNGSequence(13)
        self.func = MyFuncApprox(self.env_discrete, learning_rate=1)
        for c in self.func.state:
            self.func.state[c]['params'] = \
                jax.tree_map(jnp.zeros_like, self.func.state[c]['params'])

    def tearDown(self):
        del self.func

    def test_update_params(self):
        grads = {
            k: jax.tree_map(lambda x: x - p, v['params'])
            for (k, v), p in zip(sorted(self.func.state.items()), primes(7))}
        self.func.update_params(**grads)
        self.assertArrayAlmostEqual(
            self.func.state['body']['params']['linear']['w'], 13 * onp.ones((5, 7)))
        self.assertArrayAlmostEqual(
            self.func.state['head_v']['params']['linear']['w'], 29 * onp.ones((7, 1)))
        self.assertArrayAlmostEqual(
            self.func.state['head_q1']['params']['linear']['w'], 19 * onp.ones((21, 1)))
        self.assertArrayAlmostEqual(
            self.func.state['head_q2']['params']['linear']['w'], 23 * onp.ones((7, 3)))
        self.assertArrayAlmostEqual(
            self.func.state['head_pi']['params']['linear']['w'], 17 * onp.ones((7, 3)))

    def test_shapes_pi(self):
        func = MyFuncApprox(self.env_box)
        transition = get_transition(func.env).to_batch()
        func_body = func.apply_funcs['body']
        func_head = func.apply_funcs['head_pi']
        params_body = func.state['body']['params']
        params_head = func.state['head_pi']['params']
        state_body = func.state['body']['function_state']
        state_head = func.state['head_pi']['function_state']
        output_body, _ = \
            func_body(params_body, state_body, func.rng, transition.S, is_training=True)
        output_head, _ = \
            func_head(params_head, state_head, func.rng, output_body, is_training=True)

        self.assertIsInstance(output_head, Mapping)
        self.assertEqual(output_head['mu'].shape, (1, 3 * 5))      # flattened
        self.assertEqual(output_head['logvar'].shape, (1, 3 * 5))  # flattened

    def test_equality(self):
        func1 = FuncApprox(self.env_discrete)
        func2 = FuncApprox(self.env_discrete)
        func3 = func1
        self.assertEqual(func1, func1)
        self.assertNotEqual(func1, func2)
        self.assertEqual(func1, func3)

    def test_action_processors_discrete(self):
        func = FuncApprox(self.env_discrete)
        params_pre = func.state['action_preprocessor']['params']
        params_post = func.state['action_postprocessor']['params']
        func_pre = func.apply_funcs['action_preprocessor']
        func_post = func.apply_funcs['action_postprocessor']
        A = get_transition(func.env).to_batch().A
        X_a = func_pre(params_pre, next(self.rngs), A)
        A_prepost = func_post(params_post, next(self.rngs), X_a)
        self.assertArrayAlmostEqual(A_prepost, A)

    def test_instantiate_from_spaces(self):
        func = FuncApprox.from_spaces(
            self.env_discrete.observation_space,
            self.env_discrete.action_space)
        assert func
