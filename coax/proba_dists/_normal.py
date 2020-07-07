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

import jax.nn
import jax.random
import jax.numpy as jnp

from ._base import ProbaDist


__all__ = (
    'NormalDist',
)


class NormalDist(ProbaDist):
    r"""

    A differentiable normal distribution.

    The input ``dist_params`` to each of the functions is expected to be of the
    form:

    .. code:: python

        dist_params = {'mu': array([...]), 'logvar': array([...])}

    which represent the (conditional) distribution parameters. Here, ``mu`` is
    the mean :math:`\mu` and ``logvar`` is the log-variance
    :math:`\log(\sigma^2)`.

    """
    def __init__(self):
        self._init_funcs()

    def _init_funcs(self):
        log_2pi = 1.8378770664093453  # log(2π)

        def sample(dist_params, rng):
            mu, logvar = dist_params['mu'], dist_params['logvar']
            z = jax.random.normal(rng, mu.shape)
            return mu + z * jnp.exp(logvar / 2)

        def mode(dist_params):
            return dist_params['mu']

        def log_proba(dist_params, x):
            mu, logvar = dist_params['mu'], dist_params['logvar']
            assert mu.ndim == logvar.ndim == 2  # check if flattened
            n = logvar.shape[-1]
            log_det_var = jnp.sum(logvar, axis=-1)  # log(det(M)) = tr(log(M))
            if x.ndim == mu.ndim == logvar.ndim == 1:
                quad = jnp.dot(jnp.square(x - mu), jnp.exp(-logvar))
            else:
                assert x.ndim == mu.ndim == logvar.ndim == 2
                quad = jnp.einsum(
                    'ij,ij->i', jnp.square(x - mu), jnp.exp(-logvar))
            return -0.5 * (n * log_2pi + log_det_var + quad)

        def entropy(dist_params):
            logvar = dist_params['logvar']
            assert logvar.ndim == 2  # check if flattened
            log_det_var = jnp.sum(logvar, axis=-1)  # log(det(M)) = tr(log(M))
            n = logvar.shape[-1]
            return 0.5 * (n * log_2pi + log_det_var + n)

        def cross_entropy(dist_params_p, dist_params_q):
            m1, log_v1 = dist_params_p['mu'], dist_params_p['logvar']
            m2, log_v2 = dist_params_q['mu'], dist_params_q['logvar']
            v1 = jnp.exp(log_v1)
            v2_inv = jnp.exp(-log_v2)
            log_det_v2 = jnp.sum(log_v2, axis=-1)  # log(det(M)) = tr(log(M))
            n = m1.shape[-1]
            assert n == m2.shape[-1] == log_v1.shape[-1] == log_v2.shape[-1]
            if m2.ndim == log_v2.ndim == 1:
                quad = jnp.dot(v1 + jnp.square(m1 - m2), v2_inv)
            else:
                assert m2.ndim == log_v2.ndim == 2
                quad = jnp.einsum('ij,ij->i', v1 + jnp.square(m1 - m2), v2_inv)
            return 0.5 * (n * log_2pi + log_det_v2 + quad)

        def kl_divergence(dist_params_p, dist_params_q):
            m1, log_v1 = dist_params_p['mu'], dist_params_p['logvar']
            m2, log_v2 = dist_params_q['mu'], dist_params_q['logvar']
            assert m1.ndim == log_v1.ndim and m1.ndim in (1, 2)  # check flat
            assert m2.ndim == log_v2.ndim and m2.ndim in (1, 2)  # check flat
            v1 = jnp.exp(log_v1)
            v2_inv = jnp.exp(-log_v2)
            log_det_v1 = jnp.sum(log_v1, axis=-1)  # log(det(M)) = tr(log(M))
            log_det_v2 = jnp.sum(log_v2, axis=-1)  # log(det(M)) = tr(log(M))
            n = m1.shape[-1]
            assert n == m2.shape[-1] == log_v1.shape[-1] == log_v2.shape[-1]
            if m2.ndim == log_v2.ndim == 1:
                quad = jnp.dot(v1 + jnp.square(m1 - m2), v2_inv)
            else:
                assert m2.ndim == log_v2.ndim == 2
                quad = jnp.einsum('ij,ij->i', v1 + jnp.square(m1 - m2), v2_inv)
            return 0.5 * (log_det_v2 - log_det_v1 + quad - n)

        self._sample_func = jax.jit(sample)
        self._mode_func = jax.jit(mode)
        self._log_proba_func = jax.jit(log_proba)
        self._entropy_func = jax.jit(entropy)
        self._cross_entropy_func = jax.jit(cross_entropy)
        self._kl_divergence_func = jax.jit(kl_divergence)

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates using the
        reparametrization trick, i.e. :math:`x\sim\mathcal{N}(\mu,\sigma^2)` is
        implemented as

        .. math::

            \varepsilon\ &\sim\ \mathcal{N}(0,1) \\
            x\ &=\ \mu + \sigma\,\varepsilon


        Parameters
        ----------
        dist_params : pytree with ndarray leaves

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
        distribution, in this case simply :math:`\mu`.


        Parameters
        ----------
        dist_params : pytree with ndarray leaves

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
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

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
            \ =\ \frac12\left( \log(2\pi\sigma^2) + 1\right)



        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

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
            \ =\ \frac12\left(
                \log(2\pi\sigma_q^2)
                + \frac{(\mu_p-\mu_q)^2+\sigma_p^2}{\sigma_q^2}
            \right)

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

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
            \ =\ \frac12\left(
                \log(\sigma_q^2) - \log(\sigma_p^2)
                + \frac{(\mu_p-\mu_q)^2+\sigma_p^2}{\sigma_q^2}
                - 1
            \right)

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution
            :math:`q`.

        """
        return self._kl_divergence_func

    @staticmethod
    def default_priors(shape):
        r"""

        The default distribution parameters:

        .. code::

            {'mu': zeros(shape), 'logvar': zeros(shape)}

        Parameters
        ----------
        shape : tuple of ints

            The shape of the distribution parameters.

        Returns
        -------
        dist_params_prior : pytree with ndarray leaves

            The distribution parameters that represent the default priors.

        """
        return {'mu': jnp.zeros(shape=shape), 'logvar': jnp.zeros(shape=shape)}
