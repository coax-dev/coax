import warnings

import jax
import jax.numpy as jnp
import numpy as onp
from gym.spaces import Box

from ..utils import clipped_logit, jit
from ._base import BaseProbaDist


__all__ = (
    'NormalDist',
)


class NormalDist(BaseProbaDist):
    r"""

    A differentiable normal distribution.

    The input ``dist_params`` to each of the functions is expected to be of the form:

    .. code:: python

        dist_params = {'mu': array([...]), 'logvar': array([...])}

    which represent the (conditional) distribution parameters. Here, ``mu`` is the mean :math:`\mu`
    and ``logvar`` is the log-variance :math:`\log(\sigma^2)`.

    Parameters
    ----------
    space : gym.spaces.Box

        The gym-style space that specifies the domain of the distribution.

    clip_box : pair of floats, optional

        The range of values to allow for *clean* (compact) variates. This is mainly to ensure
        reasonable values when one or more dimensions of the Box space have very large ranges, while
        in reality only a small part of that range is occupied.

    clip_reals : pair of floats, optional

        The range of values to allow for *raw* (decompactified) variates, the *reals*, used
        internally. This range is set for numeric stability. Namely, the :attr:`postprocess_variate`
        method compactifies the reals to a closed interval (Box) by applying a logistic sigmoid.
        Setting a finite range for :code:`clip_reals` ensures that the sigmoid doesn't fully
        saturate.

    """
    def __init__(self, space, clip_box=(-256., 256.), clip_reals=(-30., 30.)):
        if not isinstance(space, Box):
            raise TypeError(f"{self.__class__.__name__} can only be defined over Box spaces")

        super().__init__(space)

        self.clip_box = clip_box
        self.clip_reals = clip_reals

        self._low = onp.maximum(onp.expand_dims(self.space.low, axis=0), self.clip_box[0])
        self._high = onp.minimum(onp.expand_dims(self.space.high, axis=0), self.clip_box[1])
        onp.testing.assert_array_less(
            self._low, self._high,
            "Box clipping resulted in inconsistent boundaries: "
            f"low={self._low}, high={self._high}; please specify proper clipping values, "
            "e.g. NormalDist(space, clip_box=(-1000., 1000.))")
        if onp.any(self._low > self.space.low) or onp.any(self._high < self.space.high):
            with onp.printoptions(precision=1):
                warnings.warn(
                    f"one or more dimensions of Box(low={self.space.low}, high={self.space.high}) "
                    f"will be clipped to Box(low={self._low[0]}, high={self._high[0]})")

        log_2pi = onp.asarray(1.8378770664093453)  # abbreviation

        def check_shape(x, name, flatten):
            if not isinstance(x, jnp.ndarray):
                raise TypeError(f"expected an jax.numpy.ndarray, got: {type(x)}")
            if not (x.ndim == len(space.shape) + 1 and x.shape[1:] == space.shape):
                expected = ', '.join(f'{i:d}' for i in space.shape)
                raise ValueError(f"expected {name}.shape: (?, {expected}), got: {x.shape}")
            if flatten:
                x = x.reshape(x.shape[0], -1)  # batch-flatten
            return x

        def sample(dist_params, rng):
            mu = check_shape(dist_params['mu'], name='mu', flatten=True)
            logvar = check_shape(dist_params['logvar'], name='logvar', flatten=True)

            X = mu + jnp.exp(logvar / 2) * jax.random.normal(rng, mu.shape)
            return X.reshape(-1, *self.space.shape)

        def mean(dist_params):
            return check_shape(dist_params['mu'], name='mu', flatten=False)

        def mode(dist_params):
            return mean(dist_params)

        def log_proba(dist_params, X):
            X = check_shape(X, name='X', flatten=True)
            mu = check_shape(dist_params['mu'], name='mu', flatten=True)
            logvar = check_shape(dist_params['logvar'], name='logvar', flatten=True)

            n = logvar.shape[-1]
            logdetvar = jnp.sum(logvar, axis=-1)  # log(det(M)) = tr(log(M))
            quadratic = jnp.einsum('ij,ij->i', jnp.square(X - mu), jnp.exp(-logvar))
            return -0.5 * (n * log_2pi + logdetvar + quadratic)

        def entropy(dist_params):
            logvar = check_shape(dist_params['logvar'], name='logvar', flatten=True)

            assert logvar.ndim == 2  # check if flattened
            logdetvar = jnp.sum(logvar, axis=-1)  # log(det(M)) = tr(log(M))
            n = logvar.shape[-1]
            return 0.5 * (n * log_2pi + logdetvar + n)

        def cross_entropy(dist_params_p, dist_params_q):
            mu1 = check_shape(dist_params_p['mu'], name='mu_p', flatten=True)
            mu2 = check_shape(dist_params_q['mu'], name='mu_q', flatten=True)
            logvar1 = check_shape(dist_params_p['logvar'], name='logvar_p', flatten=True)
            logvar2 = check_shape(dist_params_q['logvar'], name='logvar_q', flatten=True)

            n = mu1.shape[-1]
            assert n == mu2.shape[-1] == logvar1.shape[-1] == logvar2.shape[-1]

            var1 = jnp.exp(logvar1)
            var2_inv = jnp.exp(-logvar2)
            logdetvar2 = jnp.sum(logvar2, axis=-1)  # log(det(M)) = tr(log(M))
            quadratic = jnp.einsum('ij,ij->i', var1 + jnp.square(mu1 - mu2), var2_inv)
            return 0.5 * (n * log_2pi + logdetvar2 + quadratic)

        def kl_divergence(dist_params_p, dist_params_q):
            mu1 = check_shape(dist_params_p['mu'], name='mu_p', flatten=True)
            mu2 = check_shape(dist_params_q['mu'], name='mu_q', flatten=True)
            logvar1 = check_shape(dist_params_p['logvar'], name='logvar_p', flatten=True)
            logvar2 = check_shape(dist_params_q['logvar'], name='logvar_q', flatten=True)

            n = mu1.shape[-1]
            assert n == mu2.shape[-1] == logvar1.shape[-1] == logvar2.shape[-1]

            var1 = jnp.exp(logvar1)
            var2_inv = jnp.exp(-logvar2)
            logdetvar1 = jnp.sum(logvar1, axis=-1)  # log(det(M)) = tr(log(M))
            logdetvar2 = jnp.sum(logvar2, axis=-1)  # log(det(M)) = tr(log(M))
            quadratic = jnp.einsum('ij,ij->i', var1 + jnp.square(mu1 - mu2), var2_inv)
            return 0.5 * (logdetvar2 - logdetvar1 + quadratic - n)

        def affine_transform_func(dist_params, scale, shift, value_transform=None):
            if value_transform is None:
                f = f_inv = lambda x: x
            else:
                f, f_inv = value_transform
            mu = check_shape(dist_params['mu'], name='mu', flatten=False)
            logvar = check_shape(dist_params['logvar'], name='logvar', flatten=False)
            var_new = f(f_inv(jnp.exp(logvar)) * jnp.square(scale))
            return {'mu': f(f_inv(mu) + shift), 'logvar': jnp.log(var_new)}

        self._sample_func = jit(sample)
        self._mean_func = jit(mean)
        self._mode_func = jit(mode)
        self._log_proba_func = jit(log_proba)
        self._entropy_func = jit(entropy)
        self._cross_entropy_func = jit(cross_entropy)
        self._kl_divergence_func = jit(kl_divergence)
        self._affine_transform_func = jit(affine_transform_func, static_argnums=(3,))

    @property
    def default_priors(self):
        shape = (1, *self.space.shape)  # include batch axis
        return {'mu': jnp.zeros(shape), 'logvar': jnp.zeros(shape)}

    def preprocess_variate(self, rng, X):
        X = jnp.asarray(X, dtype=self.space.dtype)                     # ensure ndarray
        X = jnp.reshape(X, (-1, *self.space.shape))                    # ensure batch axis
        X = jnp.clip(X, self._low, self._high)                         # clip to be safe
        X = clipped_logit((X - self._low) / (self._high - self._low))  # closed intervals -> reals
        return X

    def postprocess_variate(self, rng, X, index=0, batch_mode=False):
        X = jnp.asarray(X, dtype=self.space.dtype)                    # ensure ndarray
        X = jnp.reshape(X, (-1, *self.space.shape))                   # ensure correct shape
        X = jnp.clip(X, *self.clip_reals)                             # clip for stability
        X = self._low + (self._high - self._low) * jax.nn.sigmoid(X)  # reals -> closed interval
        return X if batch_mode else onp.asanyarray(X[index])

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates using the reparametrization
        trick, i.e. :math:`x\sim\mathcal{N}(\mu,\sigma^2)` is implemented as

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
    def mean(self):
        r"""

        JIT-compiled functions that generates differentiable means of the distribution, in this case
        simply :math:`\mu`.

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
        return self._entropy_func

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
        return self._cross_entropy_func

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
        return self._kl_divergence_func
