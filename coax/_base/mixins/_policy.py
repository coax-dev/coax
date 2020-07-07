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

from abc import ABC, abstractmethod

from ._random_state import RandomStateMixin

__all__ = (
    'PolicyMixin',
)


class PolicyMixin(ABC, RandomStateMixin):

    @abstractmethod
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

            The log-propensity :math:`\log\pi(a|s)`. This is only returned if
            we set ``return_logp=True``.

        """

    @abstractmethod
    def greedy(self, s):
        r"""

        Get the action greedily :math:`a=\arg\max_a\pi(a|s)`..

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        """

    @abstractmethod
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

            The distribution parameters of :math:`\pi(.|s)`. For instance, for
            a categorical distribution this would be ``Params({'logits':
            array([...])})``. For a normal distribution it is ``Params({'mu':
            array([...]), 'logvar': array([...])})``

        """
