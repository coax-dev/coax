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


import gym

from .._base.test_case import TestCase
from .func_approx import FuncApprox
from .value_q import Q
from .policy_q import EpsilonGreedy, BoltzmannPolicy


class TestValueBasedPolicies(TestCase):
    def setUp(self):
        self.env = gym.make('FrozenLakeNonSlippery-v0')
        self.func = FuncApprox(self.env, learning_rate=0.2)
        self.q = Q(self.func)

    def tearDown(self):
        del self.env, self.func, self.q

    def test_epsilon_greedy(self):
        pi = EpsilonGreedy(self.q, epsilon=0.1)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_epsilon_greedy_greedy(self):
        pi = EpsilonGreedy(self.q, epsilon=0.1)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi.greedy(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_boltzmann_policy(self):
        pi = BoltzmannPolicy(self.q, tau=1.0)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_boltzmann_policy_greedy(self):
        pi = BoltzmannPolicy(self.q, tau=1.0)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi.greedy(s)
            s, r, done, info = self.env.step(a)
            if done:
                break
