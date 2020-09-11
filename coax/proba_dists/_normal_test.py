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
import jax
import haiku as hk

from .._base.test_case import TestCase
from ._normal import NormalDist


class TestNormalDist(TestCase):
    decimal = 5

    def setUp(self):
        self.rngs = hk.PRNGSequence(13)

    def tearDown(self):
        del self.rngs

    def test_kl_divergence(self):
        dist = NormalDist(gym.spaces.Box(low=0, high=1, shape=(7,)))
        params_p = {
            'mu': jax.random.normal(next(self.rngs), shape=(3, 7)),
            'logvar': jax.random.normal(next(self.rngs), shape=(3, 7))}
        params_q = {
            'mu': jax.random.normal(next(self.rngs), shape=(3, 7)),
            'logvar': jax.random.normal(next(self.rngs), shape=(3, 7))}
        # params_q = {k: v + 0.001 for k, v in params_p.items()}

        kl_div_direct = dist.kl_divergence(params_p, params_q)
        kl_div_from_ce = dist.cross_entropy(params_p, params_q) - dist.entropy(params_p)
        self.assertArrayAlmostEqual(kl_div_direct, kl_div_from_ce)

    def test_box_clip(self):
        msg = (
            r"one or more dimensions of Box\(low=.*, high=.*\) "
            r"will be clipped to Box\(low=.*, high=.*\)"
        )
        with self.assertWarnsRegex(UserWarning, msg):
            dist = NormalDist(gym.spaces.Box(low=-1000, high=10000000, shape=(1,)))

        self.assertGreater(dist._low[0, 0], dist.space.low[0])
        self.assertLess(dist._high[0, 0], dist.space.high[0])
