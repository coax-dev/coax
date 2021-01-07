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
from ._base import Regularizer


class EntropyRegularizer(Regularizer):
    r"""

    Policy regularization term based on the entropy of the policy.

    The regularization term is to be added to the loss function:

    .. math::

        \text{loss}(\theta; s,a)\ =\ -J(\theta; s,a) - \beta\,H[\pi_\theta(.|s)]

    where :math:`J(\theta)` is the bare policy objective.

    Parameters
    ----------
    f : stochastic function approximator

        The stochastic function approximator (e.g. :class:`coax.Policy`) to regularize.

    beta : non-negative float

        The coefficient that determines the strength of the overall regularization term.

    """
    def __init__(self, f, beta=0.001):
        super().__init__(f)
        self.beta = beta

        def function(dist_params, beta):
            entropy = self.f.proba_dist.entropy(dist_params)
            return -beta * entropy

        def metrics(dist_params, beta):
            entropy = self.f.proba_dist.entropy(dist_params)
            return {
                'EntropyRegularizer/beta': beta,
                'EntropyRegularizer/entropy': jnp.mean(entropy)}

        self._function = jit(function)
        self._metrics_func = jit(metrics)

    @property
    def hyperparams(self):
        return {'beta': self.beta}

    @property
    def function(self):
        r"""

        JIT-compiled function that returns the values for the regularization term.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the (conditional) probability distribution.

        beta : non-negative float

            The coefficient that determines the strength of the overall regularization term.

        """
        return self._function

    @property
    def metrics_func(self):
        r"""

        JIT-compiled function that returns the performance metrics for the regularization term.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the (conditional) probability distribution.

        beta : non-negative float

            The coefficient that determines the strength of the overall regularization term.

        """
        return self._metrics_func
