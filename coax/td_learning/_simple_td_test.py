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
from .._core.value_v import V
from ..utils import get_transition
from ._simple_td import SimpleTD


class TestSimpleTD(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition(self.env_discrete).to_batch()
        self.transition_boxspace = get_transition(self.env_boxspace).to_batch()

    def test_update_discrete(self):
        env = self.env_discrete
        func_v = self.func_v

        v = V(func_v, env.observation_space)
        v_targ = v.copy()
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0))

        params = deepcopy(v.params)
        function_state = deepcopy(v.function_state)

        updater.update(self.transition_discrete)

        self.assertPytreeNotEqual(params, v.params)
        self.assertPytreeNotEqual(function_state, v.function_state)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func_v = self.func_v

        v = V(func_v, env.observation_space)
        v_targ = v.copy()
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0))

        params = deepcopy(v.params)
        function_state = deepcopy(v.function_state)

        updater.update(self.transition_boxspace)

        self.assertPytreeNotEqual(params, v.params)
        self.assertPytreeNotEqual(function_state, v.function_state)
