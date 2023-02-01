import jax
import jax.numpy as jnp
import numpy as onp

from ._base import BaseProbaDist
from ._normal import NormalDist


class SquashedNormalDist(BaseProbaDist):
    r"""

    A differentiable squashed normal distribution.

    The input ``dist_params`` to each of the functions is expected to be of the form:

    .. code:: python

        dist_params = {'mu': array([...]), 'logvar': array([...])}

    which represent the (conditional) distribution parameters. Here, ``mu`` is the mean :math:`\mu`
    and ``logvar`` is the log-variance :math:`\log(\sigma^2)`.

    Parameters
    ----------
    space : gymnasium.spaces.Box

        The gymnasium-style space that specifies the domain of the distribution.

    clip_logvar : pair of floats, optional

        The range of values to allow for the log-variance of the distribution.

    """

    def __init__(self, space, clip_logvar=None):
        super().__init__(space)

        self._normal_dist = NormalDist(space=space, clip_logvar=clip_logvar)
        self._scale = (space.high - space.low) / 2.0
        self._offset = (space.high + space.low) / 2.0

        def sample(dist_params, rng):
            X = self._normal_dist.sample(dist_params, rng)
            return jnp.tanh(X) * self._scale + self._offset

        def mean(dist_params):
            mu = self._normal_dist.mean(dist_params)
            return jnp.tanh(mu) * self._scale + self._offset

        def mode(dist_params):
            return mean(dist_params)

        arctanh_eps = 1e-7  # avoid arctanh(1) = acrtanh(-1) = inf

        def log_proba(dist_params, X):
            X = jnp.arctanh(jnp.clip(X, a_min=-1.0 + arctanh_eps, a_max=1.0 - arctanh_eps))
            logp = self._normal_dist.log_proba(dist_params, X)
            return logp - jnp.sum(2 * (jnp.log(2) - X - jnp.log(1 + jnp.exp(-2 * X))), axis=-1)

        self._sample_func = jax.jit(sample)
        self._mean_func = jax.jit(mean)
        self._mode_func = jax.jit(mode)
        self._log_proba_func = jax.jit(log_proba)
        self._affine_transform_func = self._normal_dist.affine_transform

    @property
    def default_priors(self):
        return self._normal_dist.default_priors

    def preprocess_variate(self, rng, X):
        X = jnp.asarray(X, dtype=self.space.dtype)                     # ensure ndarray
        X = jnp.reshape(X, (-1, *self.space.shape))                    # ensure batch axis
        return X

    def postprocess_variate(self, rng, X, index=0, batch_mode=False):
        X = jnp.asarray(X, dtype=self.space.dtype)                    # ensure ndarray
        X = jnp.reshape(X, (-1, *self.space.shape))                   # ensure correct shape
        return X if batch_mode else onp.asanyarray(X[index])

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates using the reparametrization
        trick, i.e. :math:`x\sim\tanh(\mathcal{N}(\mu,\sigma^2))` is implemented as

        .. math::

            \varepsilon\ &\sim\ \mathcal{N}(0,1) \\
            x\ &=\ \tanh(\mu + \sigma\,\varepsilon)

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
    def mean(self):
        r"""

        JIT-compiled functions that generates differentiable means of the distribution, in this case
        simply :math:`\tanh(\mu)`.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._mean_func

    @property
    def mode(self):
        r"""

        JIT-compiled functions that generates differentiable modes of the distribution, which for a
        normal distribution is the same as the :attr:`mean`.

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

            A batch of variates, e.g. a batch of actions :math:`a` collected from experience.

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
        return self._normal_dist.entropy

    @property
    def cross_entropy(self):
        r"""

        JIT-compiled function that computes the cross-entropy of a distribution :math:`q` relative
        to another categorical distribution :math:`p`:

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

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._normal_dist.cross_entropy

    @property
    def kl_divergence(self):
        r"""

        JIT-compiled function that computes the Kullback-Leibler divergence of a categorical
        distribution :math:`q` relative to another distribution :math:`p`:

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

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._normal_dist.kl_divergence
