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

import jax
import haiku as hk

from ..utils import batch_to_single, single_to_batch

__all__ = (
    'PolicyMixin',
)


class PolicyMixin:
    """ Mix-in class for common functionality shared by policies. """

    def __call__(self, s, return_logp=False):
        r"""

        Sample an action :math:`a\sim\pi(.|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        return_logp : bool, optional

            Whether to return the log-propensity :math:`\log\pi(a|s)`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        logp : float, optional

            The log-propensity :math:`\log\pi(a|s)`. This is only returned if we set
            ``return_logp=True``.

        """
        S = single_to_batch(s)
        A, logP = self.sample_func(self.params, self.function_state, self.rng, S)
        a = self.proba_dist.postprocess_variate(A)
        return (a, batch_to_single(logP)) if return_logp else a

    def greedy(self, s):
        r"""

        Sample a greedy action :math:`a=\arg\max_a\pi(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        """
        S = single_to_batch(s)
        A = self.mode_func(self.params, self.function_state, self.rng, S)
        a = self.proba_dist.postprocess_variate(A)
        return a

    def dist_params(self, s):
        r"""

        Get the conditional distribution parameters of :math:`\pi(.|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        dist_params : Params

            The distribution parameters of :math:`\pi(.|s)`.

        """
        S = single_to_batch(s)
        dist_params, _ = self.function(self.params, self.function_state, self.rng, S, False)
        return batch_to_single(dist_params)

    @property
    def sample_func(self):
        r"""

        The function that is used for sampling *random* actions, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.sample_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_sample_func'):
            def func(params, state, rng, S):
                rngs = hk.PRNGSequence(rng)
                dist_params, _ = self.function(params, state, next(rngs), S, False)
                A = self.proba_dist.sample(dist_params, next(rngs))
                logP = self.proba_dist.log_proba(dist_params, A)
                return A, logP
            self._sample_func = jax.jit(func)
        return self._sample_func

    @property
    def mode_func(self):
        r"""

        The function that is used for sampling *greedy* actions, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.mode_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_mode_func'):
            def func(params, state, rng, S):
                dist_params, _ = self.function(params, state, rng, S, False)
                A = self.proba_dist.mode(dist_params)
                return A
            self._mode_func = jax.jit(func)
        return self._mode_func
