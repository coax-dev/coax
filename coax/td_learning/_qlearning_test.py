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

from optax import sgd

from .._base.test_case import TestCase
from .._core.q import Q
from .._core.policy import Policy
from ..utils import get_transition_batch
from ._qlearning import QLearning


class TestQLearning(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition_batch(self.env_discrete, random_seed=42)
        self.transition_boxspace = get_transition_batch(self.env_boxspace, random_seed=42)

    def test_update_discrete_type1(self):
        env = self.env_discrete
        func_q = self.func_q_type1

        q = Q(func_q, env)
        q_targ = q.copy()
        updater = QLearning(q, q_targ=q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_discrete)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_discrete_type2(self):
        env = self.env_discrete
        func_q = self.func_q_type2

        q = Q(func_q, env)
        q_targ = q.copy()
        updater = QLearning(q, q_targ=q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_discrete)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func_q = self.func_q_type1
        func_pi = self.func_pi_boxspace

        q = Q(func_q, env)
        pi = Policy(func_pi, env)
        q_targ = q.copy()
        updater = QLearning(q, pi, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_boxspace)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_discrete_with_pi(self):
        env = self.env_discrete
        func_q = self.func_q_type1
        func_pi = self.func_pi_discrete

        q = Q(func_q, env)
        pi = Policy(func_pi, env)
        q_targ = q.copy()

        msg = r"pi_targ is ignored, because action space is discrete"
        with self.assertWarnsRegex(UserWarning, msg):
            QLearning(q, pi, q_targ)

    def test_boxspace_without_pi(self):
        env = self.env_boxspace
        func_q = self.func_q_type1

        q = Q(func_q, env)
        q_targ = q.copy()

        msg = r"pi_targ must be provided if action space is not discrete"
        with self.assertRaisesRegex(TypeError, msg):
            QLearning(q, q_targ=q_targ)

    def test_mistake_q_for_pi(self):
        env = self.env_discrete
        func_q = self.func_q_type1

        q = Q(func_q, env)
        q_targ = q.copy()

        msg = r"pi_targ must be a Policy, got: .*"
        with self.assertRaisesRegex(TypeError, msg):
            QLearning(q, q_targ)
