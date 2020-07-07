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
import haiku as hk

from .._base.test_case import TestCase, DummyFuncApprox
from .._core.policy import Policy
from .._core.value_q import Q
from ..utils import get_transition
from ._deterministic_pg import DeterministicPG


class TestDeterministicPG(TestCase):
    margin = 1e-5

    def setUp(self):
        self.rngs = hk.PRNGSequence(13)

    def test_update_discrete(self):
        transition_batch = get_transition(self.env_discrete).to_batch()
        func = DummyFuncApprox(self.env_discrete)
        for k, v in func.state.items():
            func.state[k]['params'] = \
                jax.tree_map(lambda x: jax.random.normal(next(self.rngs), x.shape), v['params'])
        pi = Policy(func)
        q = Q(func)
        p1 = pi.batch_eval(transition_batch.S)
        print(p1)
        ddpg = DeterministicPG(pi, q)
        b1 = deepcopy(pi.func_approx.state['body']['params'])
        h1 = deepcopy(pi.func_approx.state['head_pi']['params'])
        o1 = deepcopy((
            pi.func_approx.state['action_preprocessor']['params'],
            pi.func_approx.state['state_action_combiner']['params'],
            pi.func_approx.state['head_v']['params'],
            pi.func_approx.state['head_q1']['params'],
            pi.func_approx.state['head_q2']['params'],
        ))
        # default value head is a linear layer with zero-initialized weights,
        # so we need not one but two updates to ensure that the body (which is
        # upstream from value head) receives a non-trivial update too
        ddpg.update(transition_batch)
        ddpg.update(transition_batch)
        b2 = pi.func_approx.state['body']['params']
        h2 = pi.func_approx.state['head_pi']['params']
        o2 = (
            pi.func_approx.state['action_preprocessor']['params'],
            pi.func_approx.state['state_action_combiner']['params'],
            pi.func_approx.state['head_v']['params'],
            pi.func_approx.state['head_q1']['params'],
            pi.func_approx.state['head_q2']['params'],
        )
        p2 = pi.batch_eval(transition_batch.S)
        self.assertPytreeNotEqual(h1, h2)
        self.assertPytreeNotEqual(b1, b2)
        self.assertPytreeNotEqual(p1, p2, margin=0.01)
        self.assertPytreeAlmostEqual(o1, o2)

    def test_update_box(self):
        env = self.env_box_decompactified
        transition_batch = get_transition(env).to_batch()
        print(transition_batch)
        func = DummyFuncApprox(env)
        for k, v in func.state.items():
            # keep head to its zero-initialized params, otherwise logvar change upon updating
            if k in {'head_pi', 'head_q2'}:
                continue
            func.state[k]['params'] = \
                jax.tree_map(lambda x: jax.random.normal(next(self.rngs), x.shape), v['params'])

        pi = Policy(func)
        q = Q(func)
        ddpg = DeterministicPG(pi, q)
        b1 = deepcopy(pi.func_approx.state['body']['params'])
        h1 = deepcopy(pi.func_approx.state['head_pi']['params'])
        o1 = deepcopy((
            pi.func_approx.state['action_preprocessor']['params'],
            pi.func_approx.state['state_action_combiner']['params'],
            pi.func_approx.state['head_v']['params'],
            pi.func_approx.state['head_q1']['params'],
            # pi.func_approx.state['head_q2']['params'],
        ))
        p1 = pi.batch_eval(transition_batch.S)
        # default value head is a linear layer with zero-initialized weights,
        # so we need not one but two updates to ensure that the body (which is
        # upstream from value head) receives a non-trivial update too
        ddpg.update(transition_batch)
        ddpg.update(transition_batch)
        b2 = pi.func_approx.state['body']['params']
        h2 = pi.func_approx.state['head_pi']['params']
        o2 = (
            pi.func_approx.state['action_preprocessor']['params'],
            pi.func_approx.state['state_action_combiner']['params'],
            pi.func_approx.state['head_v']['params'],
            pi.func_approx.state['head_q1']['params'],
            # pi.func_approx.state['head_q2']['params'],
        )
        p2 = pi.batch_eval(transition_batch.S)
        self.assertPytreeNotEqual(h1, h2)
        self.assertPytreeNotEqual(b1, b2)
        self.assertPytreeNotEqual(p1['mu'], p2['mu'])
        self.assertPytreeAlmostEqual(p1['logvar'], p2['logvar'])
        self.assertPytreeAlmostEqual(o1, o2)
