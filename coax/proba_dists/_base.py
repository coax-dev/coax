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


class ProbaDist(ABC):
    r"""

    Abstract base class for probability distributions. Check out
    :class:`coax.proba_dists.CategoricalDist` for a specific
    example.

    """
    __slots__ = (
        '_sample_func',
        '_mode_func',
        '_log_proba_func',
        '_entropy_func',
        '_cross_entropy_func',
        '_kl_divergence_func',
        '_default_priors_func',
    )

    @property
    def hyperparams(self):
        r""" Hyperparameters specific to this probability distribution. """
        return {}

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates.

        Parameters
        ----------
        params : pytree with ndarray leaves

            A batch of distribution parameters.

        rng : PRNGKey

            A key for seeding the pseudo-random number generator.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._sample_func

    @property
    def mode(self):
        r"""

        JIT-compiled functions that generates differentiable modes of the
        distribution.

        Parameters
        ----------
        params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._mode_func

    @property
    def log_proba(self):
        r"""

        JIT-compiled function that evaluates log-probabilities.

        Parameters
        ----------
        params : pytree with ndarray leaves

            A batch of distribution parameters of the form ``{'logits':
            ndarray}``.

        X : ndarray

            A batch of variates, e.g. a batch of actions :math:`a` collected
            from experience.

        Returns
        -------
        logP : ndarray of floats

            A batch of log-probabilities associated with the provided variates.

        """
        return self._log_proba_func

    @property
    def entropy(self):
        r"""

        JIT-compiled function that computes the entropy of the distribution.

        .. math::

            H\ =\ -\mathbb{E}_p \log p


        Parameters
        ----------
        params : pytree with ndarray leaves

            A batch of distribution parameters of the form ``{'logits':
            ndarray}``.

        Returns
        -------
        H : ndarray of floats

            A batch of entropy values.

        """
        return self._entropy_func

    @property
    def cross_entropy(self):
        r"""

        JIT-compiled function that computes the cross-entropy of a distribution
        :math:`q` relative to another categorical distribution :math:`p`:

        .. math::

            \text{CE}[p,q]\ =\ -\mathbb{E}_p \log q

        Parameters
        ----------
        params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution
            :math:`q`.

        """
        return self._cross_entropy_func

    @property
    def kl_divergence(self):
        r"""

        JIT-compiled function that computes the Kullback-Leibler divergence of
        a categorical distribution :math:`q` relative to another distribution
        :math:`p`:

        .. math::

            \text{KL}[p,q]\ = -\mathbb{E}_p \left(\log q -\log p\right)

        Parameters
        ----------
        params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution
            :math:`q`.

        """
        return self._kl_divergence_func

    @staticmethod
    @abstractmethod
    def default_priors(shape):
        r"""

        The default distribution parameters.

        Parameters
        ----------
        shape : tuple of ints

            The shape of the distribution parameters.

        Returns
        -------
        params_prior : pytree with ndarray leaves

            The distribution parameters that represent the default priors.

        """
        pass
