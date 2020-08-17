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

from .._core.policy import Policy


class PolicyRegularizer:
    r"""

    Abstract base class for policy regularizers. Check out
    :class:`coax.policy_regularizers.EntropyRegularizer` for a specific example.

    Parameters
    ----------
    pi : Policy

        The policy to be regularized.

    """
    def __init__(self, pi):
        if not isinstance(pi, Policy):
            raise TypeError(f"proba_dist must be a Policy, got {type(pi)}")

        self.pi = pi

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

            The distribution parameters of the (conditional) probability distribution
            :math:`\pi(a|s)`.

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
