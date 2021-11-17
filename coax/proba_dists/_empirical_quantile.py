import chex
import jax
import jax.numpy as jnp
from gym.spaces import Box

from ..utils import jit, isscalar
from ._base import BaseProbaDist


__all__ = (
    'EmpiricalQuantileDist',
)


class EmpiricalQuantileDist(BaseProbaDist):

    def __init__(self, num_quantiles):
        self.num_quantiles = num_quantiles
        super().__init__(Box(low=-jnp.inf, high=jnp.inf, shape=[num_quantiles]))

        def check_shape(x, name):
            if not isinstance(x, jnp.ndarray):
                raise TypeError(f"expected an jax.numpy.ndarray, got: {type(x)}")
            return x

        def mean(dist_params):
            values = check_shape(dist_params['values'], 'values')
            return jnp.mean(values, axis=-1)

        def sample(dist_params, rng):
            # bootstrapping
            values = check_shape(dist_params['values'], 'values')
            return jax.random.choice(rng, values, values.shape, replace=True)

        def log_proba(dist_params, X):
            X = check_shape(X, 'X')
            values = check_shape(dist_params['values'], 'values')
            occurrences = jnp.mean(X[None, ...] == values[..., None], axis=-1)
            return jnp.log(occurrences)

        def affine_transform(dist_params, scale, shift, value_transform=None):
            chex.assert_rank([dist_params['values'], scale, shift], [2, {0, 1}, {0, 1}])
            values = check_shape(dist_params['values'], 'values')
            quantile_fractions = check_shape(
                dist_params['quantile_fractions'], 'quantile_fractions')
            batch_size = values.shape[0]

            if isscalar(scale):
                scale = jnp.full(shape=(batch_size, 1), fill_value=jnp.squeeze(scale))
            if isscalar(shift):
                shift = jnp.full(shape=(batch_size, 1), fill_value=jnp.squeeze(shift))

            scale = jnp.reshape(scale, (batch_size, 1))
            shift = jnp.reshape(shift, (batch_size, 1))

            chex.assert_shape(values, (batch_size, self.num_quantiles))
            chex.assert_shape([scale, shift], (batch_size, 1))

            if value_transform is None:
                f = f_inv = lambda x: x
            else:
                f, f_inv = value_transform

            return {'values': f(shift + scale *
                                f_inv(values)), 'quantile_fractions': quantile_fractions}

        self._sample_func = jit(sample)
        self._mean_func = jit(mean)
        self._log_proba_func = jit(log_proba)
        self._affine_transform_func = jit(affine_transform, static_argnums=(3,))

    @property
    def default_priors(self):
        return {'values': jnp.zeros((1, self.num_quantiles)),
                'quantile_fractions': jnp.ones((1, self.num_quantiles,))}

    @property
    def sample(self):
        return self._sample_func

    @property
    def mean(self):
        return self._mean_func

    @property
    def log_proba(self):
        return self._log_proba_func
