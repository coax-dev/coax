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

import jax.numpy as jnp

from ..utils import jit
from ._entropy import EntropyRegularizer


class NStepEntropyRegularizer(EntropyRegularizer):

    def __init__(self, f, n, beta=0.001, gamma=0.99):
        super().__init__(f)
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self._gammas = jnp.power(self.gamma, jnp.arange(self.n))

        def function(dist_params, beta):
            assert len(dist_params) == 2
            entropy = sum([gamma * self.f.proba_dist.entropy(p) * (1 - d)
                           for i, (p, d, gamma) in enumerate(zip(*dist_params, self._gammas))])
            return -beta * entropy

        def metrics(dist_params, beta):
            entropy = sum([gamma * self.f.proba_dist.entropy(p) * (1 - d)
                           for i, (p, d, gamma) in enumerate(zip(*dist_params, self._gammas))])
            return {
                'EntropyRegularizer/beta': beta,
                'EntropyRegularizer/entropy': jnp.mean(entropy)}

        self._function = jit(function)
        self._metrics_func = jit(metrics)
