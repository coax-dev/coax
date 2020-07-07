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

from .._base.test_case import TestCase, DummyFuncApprox
from ..utils import get_transition
from ..wrappers import TrainMonitor
from .policy import Policy


class TestPolicy(TestCase):

    def setUp(self):
        self.funcs = {
            'discrete': DummyFuncApprox(TrainMonitor(self.env_discrete)),
            'box': DummyFuncApprox(TrainMonitor(self.env_box_decompactified))}
        self.transitions = {
            'discrete': get_transition(self.env_discrete),
            'box': get_transition(self.env_box_decompactified)}
        self.transitions_batch = {
            'discrete': get_transition(self.env_discrete).to_batch(),
            'box': get_transition(self.env_box_decompactified).to_batch()}

    def tearDown(self):
        del self.funcs, self.transitions, self.transitions_batch

    def test_batch_eval_discrete(self):
        action_type = 'discrete'
        tn = self.transitions_batch[action_type]
        pi = Policy(self.funcs[action_type])

        dist_params = pi.batch_eval(tn.S)
        self.assertIsInstance(dist_params, Mapping)
        self.assertSetEqual(set(dist_params), {'logits'})
        self.assertArrayShape(dist_params['logits'], (1, 3))
        self.assertArraySubdtypeFloat(dist_params['logits'])

    def test_batch_eval_box(self):
        action_type = 'box'
        tn = self.transitions_batch[action_type]
        pi = Policy(self.funcs[action_type])

        dist_params = pi.batch_eval(tn.S)
        self.assertIsInstance(dist_params, Mapping)
        self.assertSetEqual(set(dist_params), {'mu', 'logvar'})
        self.assertEqual(dist_params['mu'].shape, (1, 3 * 5))      # flattened
        self.assertEqual(dist_params['logvar'].shape, (1, 3 * 5))  # flattened
        self.assertArraySubdtypeFloat(dist_params['mu'])
        self.assertArraySubdtypeFloat(dist_params['logvar'])

    def test_call_discrete(self):
        action_type = 'discrete'
        tn = self.transitions[action_type]
        pi = Policy(self.funcs[action_type], random_seed=13)

        a = pi(tn.s)
        self.assertTrue(self.env_discrete.action_space.contains(a))
        self.assertEqual(a, 2)

    def test_call_box(self):
        action_type = 'box'
        tn = self.transitions[action_type]
        pi = Policy(self.funcs[action_type], random_seed=self.seed)

        a = pi(tn.s)
        print(type(a), a.shape, a.dtype)
        print(a)
        self.assertTrue(self.env_box_decompactified.action_space.contains(a))
        self.assertArrayShape(a, (3 * 5,))
        self.assertArraySubdtypeFloat(a)
