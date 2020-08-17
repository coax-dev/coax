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

from jax.experimental.optix import sgd

from .._base.test_case import TestCase
from .._core.value_q import Q
from .._core.policy import Policy
from ..utils import get_transition
from ._qlearningmode import QLearningMode


class TestQLearningMode(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition(self.env_discrete).to_batch()
        self.transition_boxspace = get_transition(self.env_boxspace).to_batch()

    def test_update_discrete(self):
        env = self.env_discrete
        func_q = self.func_q_type1
        func_pi = self.func_pi_discrete

        q = Q(func_q, env.observation_space, env.action_space)
        pi = Policy(func_pi, env.observation_space, env.action_space)
        q_targ = q.copy()
        updater = QLearningMode(q, pi, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_discrete)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func_q = self.func_q_type1
        func_pi = self.func_pi_boxspace

        q = Q(func_q, env.observation_space, env.action_space)
        pi = Policy(func_pi, env.observation_space, env.action_space)
        q_targ = q.copy()
        updater = QLearningMode(q, pi, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_boxspace)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_type2(self):
        env = self.env_discrete
        func_q = self.func_q_type2
        func_pi = self.func_pi_discrete

        q = Q(func_q, env.observation_space, env.action_space)
        pi = Policy(func_pi, env.observation_space, env.action_space)
        q_targ = q.copy()

        with self.assertRaisesRegex(TypeError, "q must be a type-1 q-function"):
            QLearningMode(q, pi, q_targ)
