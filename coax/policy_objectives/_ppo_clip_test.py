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

from .._base.test_case import TestCase, DummyFuncApprox
from .._core.policy import Policy
from ..utils import get_transition
from ._ppo_clip import PPOClip


class TestPPOClip(TestCase):
    def setUp(self):
        self.func = DummyFuncApprox(self.env_discrete, learning_rate=0.1)
        self.transition_batch = get_transition(self.env_discrete).to_batch()

    def test_update(self):
        pi = Policy(self.func)
        ppo = PPOClip(pi)
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
        ppo.update(self.transition_batch, Adv=self.transition_batch.Rn)
        ppo.update(self.transition_batch, Adv=self.transition_batch.Rn)
        b2 = pi.func_approx.state['body']['params']
        h2 = pi.func_approx.state['head_pi']['params']
        o2 = (
            pi.func_approx.state['action_preprocessor']['params'],
            pi.func_approx.state['state_action_combiner']['params'],
            pi.func_approx.state['head_v']['params'],
            pi.func_approx.state['head_q1']['params'],
            pi.func_approx.state['head_q2']['params'],
        )
        self.assertPytreeNotEqual(b1, b2)
        self.assertPytreeNotEqual(h1, h2)
        self.assertPytreeAlmostEqual(o1, o2)
