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

from itertools import combinations

import gym
import numpy as onp

from .._base.test_case import TestCase
from ._box_spaces import BoxActionsToDiscrete


class TestBoxActionsToDiscrete(TestCase):
    def test_inverse(self):
        num_bins = 100

        env = gym.make('BipedalWalker-v3')
        env = BoxActionsToDiscrete(env, num_bins)

        hi, lo = env.env.action_space.high, env.env.action_space.low
        a_orig = env.env.action_space.sample()

        # create discrete action
        a_orig_rescaled = (a_orig - lo) / (hi - lo)
        a_orig_flat = onp.ravel(a_orig_rescaled)
        a_discrete = onp.asarray(num_bins * a_orig_flat, dtype='int8')

        # reconstruct continuous action
        a_reconstructed = env._discrete_to_box(a_discrete)

        diff = onp.abs(a_reconstructed - a_orig) / (hi - lo)
        print(diff)
        self.assertTrue(onp.all(diff < 1 / num_bins))

    def test_not_all_same(self):
        env = gym.make('BipedalWalker-v3')
        env = BoxActionsToDiscrete(env, num_bins=10, random_seed=13)

        a_discrete = onp.zeros(4)
        a_box = env._discrete_to_box(a_discrete)
        print(a_box)

        for x, y in combinations(a_box, 2):
            self.assertNotAlmostEqual(x, y)
