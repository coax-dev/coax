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

import haiku as hk

from ..utils import is_stochastic, jit
from .._core.base_stochastic_func_type1 import BaseStochasticFuncType1
from .._core.base_stochastic_func_type2 import BaseStochasticFuncType2


class Regularizer:
    r"""

    Abstract base class for policy regularizers. Check out
    :class:`coax.regularizers.EntropyRegularizer` for a specific example.

    Parameters
    ----------
    f : stochastic function approximator

        The stochastic function approximator (e.g. :class:`coax.Policy`) to regularize.

    """
    def __init__(self, f):
        if not is_stochastic(f):
            raise TypeError(f"proba_dist must be a stochastic function, got {type(f)}")
        self.f = f

    @property
    def hyperparams(self):
        return {}

    @property
    def function(self):
        r"""

        JIT-compiled function that returns the values for the regularization term.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the (conditional) probability distribution.

        \*\*hyperparams

            Hyperparameters specific to the regularizer, see :attr:`hyperparams`.

        """
        return self._function

    @property
    def metrics_func(self):
        r"""

        JIT-compiled function that returns the performance metrics for the regularization term.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the (conditional) probability distribution
            :math:`\pi(a|s)`.

        \*\*hyperparams

            Hyperparameters specific to the regularizer, see :attr:`hyperparams`.

        """
        return self._metrics_func

    @property
    def batch_eval(self):
        if not hasattr(self, '_batch_eval_func'):
            def batch_eval_func(params, hyperparams, state, rng, transition_batch):
                rngs = hk.PRNGSequence(rng)
                if isinstance(self.f, BaseStochasticFuncType1):
                    S = self.f.observation_preprocessor(next(rngs), transition_batch.S)
                    A = self.f.action_preprocessor(next(rngs), transition_batch.A)
                    dist_params, _ = self.f.function(params, state, next(rngs), S, A, False)
                if isinstance(self.f, BaseStochasticFuncType2):
                    S = self.f.observation_preprocessor(next(rngs), transition_batch.S)
                    dist_params, _ = self.f.function(params, state, next(rngs), S, False)
                else:
                    raise TypeError(
                        "f must be derived from BaseStochasticFuncType1 or BaseStochasticFuncType2")
                return self.function(dist_params, **hyperparams)

            self._batch_eval_func = jit(batch_eval_func)

        return self._batch_eval_func
