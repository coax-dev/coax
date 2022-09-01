import jax
import jax.numpy as jnp
from jax.scipy.stats.beta import logpdf
import numpy as onp
from gym.spaces import Box

from ..utils import jit
from ._base import BaseProbaDist


__all__ = (
    'BetaDist',
)


class BetaDist(BaseProbaDist):
    r"""

    A differentiable beta distribution.

    The input ``dist_params`` to each of the functions is expected to be of the form:

    .. code:: python

        dist_params = {'alpha': array([...]), 'beta': array([...])}

    which represent the (conditional) distribution parameters.
    Here, ``alpha`` is the mean :math:`\mu` and ``beta`` is the log-variance :math:`\log(\sigma^2)`.

    Parameters
    ----------
    space : gym.spaces.Box

        The gym-style space that specifies the domain of the distribution.

    """

    def __init__(self, space):
        if not isinstance(space, Box):
            raise TypeError(f"{self.__class__.__name__} can only be defined over Box spaces")

        super().__init__(space)

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
            alpha = check_shape(dist_params['alpha'], name='alpha', flatten=True)
            beta = check_shape(dist_params['beta'], name='beta', flatten=True)

            X = ...  # TODO(frederik): implement

            return X.reshape(-1, *self.space.shape)

        def mean(dist_params):
            alpha = check_shape(dist_params['alpha'], name='alpha', flatten=True)
            beta = check_shape(dist_params['beta'], name='beta', flatten=True)
            return alpha / (alpha + beta)

        def mode(dist_params):
            alpha = check_shape(dist_params['alpha'], name='alpha', flatten=True)
            beta = check_shape(dist_params['beta'], name='beta', flatten=True)
            return jnp.where(alpha > 1 & beta > 1, (alpha - 1) / (alpha + beta - 2), )

        def log_proba(dist_params, X):
            alpha = check_shape(dist_params['alpha'], name='alpha', flatten=True)
            beta = check_shape(dist_params['beta'], name='beta', flatten=True)
            return logpdf(X, alpha, beta)

        def entropy(dist_params):
            logvar = check_shape(dist_params['logvar'], name='logvar', flatten=True)

            assert logvar.ndim == 2  # check if flattened
            logdetvar = jnp.sum(logvar, axis=-1)  # log(det(M)) = tr(log(M))
            n = logvar.shape[-1]
            return 0.5 * (n * log_2pi + logdetvar + n)

        def cross_entropy(dist_params_p, dist_params_q):
            pass  # TODO(frederik): implement

        def kl_divergence(dist_params_p, dist_params_q):
            pass  # TODO(frederik): implement

        def affine_transform_func(dist_params, scale, shift, value_transform=None):
            pass  # TODO(frederik): implement

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
        return {'alpha': jnp.ones(shape) * 2., 'beta': jnp.ones(shape) * 2.}

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
        return self._sample_func

    @property
    def mean(self):
        return self._mean_func

    @property
    def mode(self):
        return self._mode_func

    @property
    def log_proba(self):
        return self._log_proba_func

    @property
    def entropy(self):
        return self._entropy_func

    @property
    def cross_entropy(self):
        return self._cross_entropy_func

    @property
    def kl_divergence(self):
        return self._kl_divergence_func
