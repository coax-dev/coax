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

import jax
import jax.numpy as jnp

from .._base.test_case import TestCase, DummyFuncApprox
from .._core.value_v import V
from ..utils import get_transition
from ._value_td import ValueTD


class TestValueTD(TestCase):
    def setUp(self):
        self.func = DummyFuncApprox(self.env_discrete, learning_rate=0.91)
        self.transition_batch = get_transition(self.env_discrete).to_batch()

    def test_grads(self):
        v = V(self.func)
        v_targ = v.copy()
        value_td = ValueTD(v, v_targ)

        grads, state, metrics = value_td.grads_and_metrics(self.transition_batch)
        self.assertPytreeNotEqual(grads, jax.tree_map(jnp.zeros_like, grads))
        self.assertPytreeNotEqual(grads['head_v'], jax.tree_map(jnp.zeros_like, grads['head_v']))

        v.params = jax.tree_multimap(lambda p, g: p - 0.91 * g, v.params, grads)
        grads, state, metrics = value_td.grads_and_metrics(self.transition_batch)
        self.assertPytreeNotEqual(grads, jax.tree_map(jnp.zeros_like, grads))
        self.assertPytreeNotEqual(grads['head_v'], jax.tree_map(jnp.zeros_like, grads['head_v']))
        self.assertPytreeNotEqual(grads['body'], jax.tree_map(jnp.zeros_like, grads['body']))

    def test_update(self):
        v = V(self.func)
        v_targ = v.copy()
        value_td = ValueTD(v, v_targ)

        b1 = deepcopy(v.func_approx.state['body']['params'])
        h1 = deepcopy(v.func_approx.state['head_v']['params'])
        o1 = deepcopy((
            v.func_approx.state['action_preprocessor']['params'],
            v.func_approx.state['state_action_combiner']['params'],
            v.func_approx.state['head_pi']['params'],
            v.func_approx.state['head_q1']['params'],
            v.func_approx.state['head_q2']['params'],
        ))

        # default value head is a linear layer with zero-initialized weights, so we need not one but
        # two updates to ensure that the body (which is upstream from value head) receives a
        # non-trivial update too
        m1 = value_td.update(self.transition_batch)
        m2 = value_td.update(self.transition_batch)
        print(m1)
        print(m2)
        b2 = v.func_approx.state['body']['params']
        h2 = v.func_approx.state['head_v']['params']
        o2 = (
            v.func_approx.state['action_preprocessor']['params'],
            v.func_approx.state['state_action_combiner']['params'],
            v.func_approx.state['head_pi']['params'],
            v.func_approx.state['head_q1']['params'],
            v.func_approx.state['head_q2']['params'],
        )
        self.assertPytreeNotEqual(h1, h2)
        self.assertPytreeNotEqual(b1, b2)
        self.assertPytreeAlmostEqual(o1, o2)
