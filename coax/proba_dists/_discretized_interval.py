import jax
import jax.numpy as jnp
import numpy as onp
import chex
from gym.spaces import Box, Discrete

from ..utils import isscalar, jit
from ._categorical import CategoricalDist


__all__ = (
    'DiscretizedIntervalDist',
)


class DiscretizedIntervalDist(CategoricalDist):
    r"""

    A categorical distribution over a discretized interval.

    The input ``dist_params`` to each of the functions is expected to be of the form:

    .. code:: python

        dist_params = {'logits': array([...])}

    which represent the (conditional) distribution parameters. The ``logits``, denoted
    :math:`z\in\mathbb{R}^n`, are related to the categorical distribution parameters
    :math:`p\in\Delta^n` via a softmax:

    .. math::

        p_k\ =\ \text{softmax}_k(z)\ =\ \frac{\text{e}^{z_k}}{\sum_j\text{e}^{z_j}}


    Parameters
    ----------
    space : gym.spaces.Box

        The gym-style space that specifies the domain of the distribution. The shape of the Box must
        have :code:`prod(shape) == 1`, i.e. a single interval.

    num_bins : int, optional

        The number of equal-sized bins used in the discretization.

    gumbel_softmax_tau : positive float, optional

        The parameter :math:`\tau` specifies the sharpness of the Gumbel-softmax sampling (see
        :func:`sample` method below). A good value for :math:`\tau` balances the trade-off between
        getting proper deterministic variates (i.e. one-hot vectors) versus getting smooth
        differentiable variates.

    """
    __slots__ = (*CategoricalDist.__slots__, '__space_orig', '__low', '__high', '__atoms')

    def __init__(self, space, num_bins=20, gumbel_softmax_tau=0.2):
        if not isinstance(space, Box):
            raise TypeError(f"{self.__class__.__name__} can only be defined over Box spaces")
        if onp.prod(space.shape) > 1:
            raise TypeError(f"{self.__class__.__name__} can only be defined a single interval")

        super().__init__(space=Discrete(num_bins), gumbel_softmax_tau=gumbel_softmax_tau)
        self.__space_orig = space
        self.__low = low = float(space.low)
        self.__high = high = float(space.high)
        self.__atoms = low + (jnp.arange(num_bins) + 0.5) * (high - low) / num_bins

        def affine_transform(dist_params, scale, shift, value_transform=None):
            """ implements the "Categorical Algorithm" from https://arxiv.org/abs/1707.06887 """

            # check inputs
            chex.assert_rank([dist_params['logits'], scale, shift], [2, {0, 1}, {0, 1}])
            p = jax.nn.softmax(dist_params['logits'])
            batch_size = p.shape[0]

            if isscalar(scale):
                scale = jnp.full(shape=(batch_size,), fill_value=jnp.squeeze(scale))
            if isscalar(shift):
                shift = jnp.full(shape=(batch_size,), fill_value=jnp.squeeze(shift))

            chex.assert_shape(p, (batch_size, self.num_bins))
            chex.assert_shape([scale, shift], (batch_size,))

            if value_transform is None:
                f = f_inv = lambda x: x
            else:
                f, f_inv = value_transform

            # variable names correspond to those defined in: https://arxiv.org/abs/1707.06887
            z = self.__atoms
            Vmin, Vmax, Δz = z[0], z[-1], z[1] - z[0]
            Tz = f(jax.vmap(jnp.add)(jnp.outer(scale, f_inv(z)), shift))
            Tz = jnp.clip(Tz, Vmin, Vmax)  # keep values in valid range
            chex.assert_shape(Tz, (batch_size, self.num_bins))

            b = (Tz - Vmin) / Δz                            # float in [0, num_bins - 1]
            l = jnp.floor(b).astype('int32')  # noqa: E741   # int in {0, 1, ..., num_bins - 1}
            u = jnp.ceil(b).astype('int32')                  # int in {0, 1, ..., num_bins - 1}
            chex.assert_shape([p, b, l, u], (batch_size, self.num_bins))

            m = jnp.zeros_like(p)
            i = jnp.expand_dims(jnp.arange(batch_size), axis=1)   # batch index
            m = m.at[(i, l)].add(p * (u - b), indices_are_sorted=True)
            m = m.at[(i, u)].add(p * (b - l), indices_are_sorted=True)
            m = m.at[(i, l)].add(p * (l == u), indices_are_sorted=True)
            # chex.assert_tree_all_close(jnp.sum(m, axis=1), jnp.ones(batch_size), rtol=1e-6)

            # # The above index trickery is equivalent to:
            # m_alt = onp.zeros((batch_size, self.num_bins))
            # for i in range(batch_size):
            #     for j in range(self.num_bins):
            #         if l[i, j] == u[i, j]:
            #             m_alt[i, l[i, j]] += p[i, j]  # don't split if b[i, j] is an integer
            #         else:
            #             m_alt[i, l[i, j]] += p[i, j] * (u[i, j] - b[i, j])
            #             m_alt[i, u[i, j]] += p[i, j] * (b[i, j] - l[i, j])
            # chex.assert_tree_all_close(m, m_alt, rtol=1e-6)
            return {'logits': jnp.log(jnp.maximum(m, 1e-16))}

        self._affine_transform_func = jit(affine_transform, static_argnums=(3,))

    @property
    def space_orig(self):
        return self.__space_orig

    @property
    def low(self):
        return self.__low

    @property
    def high(self):
        return self.__high

    @property
    def num_bins(self):
        return self.space.n

    @property
    def atoms(self):
        return self.__atoms.copy()

    def preprocess_variate(self, rng, X):
        X = jnp.asarray(X)
        assert X.ndim <= 1, f"unexpected X.shape: {X.shape}"
        assert jnp.issubdtype(X.dtype, jnp.integer), f"expected an integer dtype, got {X.dtype}"
        low, high = float(self.space_orig.low), float(self.space_orig.high)
        return jax.nn.one_hot(jnp.floor((X - low) * self.num_bins / (high - low)), self.num_bins)

    def postprocess_variate(self, rng, X, index=0, batch_mode=False):
        # map almost-one-hot vectors to bin-indices (ints)
        chex.assert_rank(X, {2, 3})
        assert X.shape[-1] == self.num_bins

        # map bin-probabilities to real values
        X = jnp.dot(X, self.__atoms)
        chex.assert_rank(X, {1, 2})

        return X if batch_mode else X[index]
