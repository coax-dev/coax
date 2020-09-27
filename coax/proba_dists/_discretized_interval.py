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
import jax.numpy as jnp
import numpy as onp
from gym.spaces import Box, Discrete

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
    __slots__ = (*CategoricalDist.__slots__, '_space_orig')

    def __init__(self, space, num_bins=20, gumbel_softmax_tau=0.2):
        if not isinstance(space, Box):
            raise TypeError(f"{self.__class__.__name__} can only be defined over Box spaces")
        if onp.prod(space.shape) > 1:
            raise TypeError(f"{self.__class__.__name__} can only be defined a single interval")

        self._space_orig = space
        super().__init__(space=Discrete(num_bins), gumbel_softmax_tau=gumbel_softmax_tau)

    @property
    def space_orig(self):
        return self._space_orig

    @property
    def num_bins(self):
        return self.space.n

    def preprocess_variate(self, rng, X):
        X = jnp.asarray(X)
        assert X.ndim <= 1, f"unexpected X.shape: {X.shape}"
        assert jnp.issubdtype(X.dtype, jnp.integer), f"expected an integer dtype, got {X.dtype}"
        low, high = float(self.space_orig.low), float(self.space_orig.high)
        return jax.nn.one_hot(jnp.floor((X - low) * self.num_bins / (high - low)), self.num_bins)

    def postprocess_variate(self, rng, X, index=0, batch_mode=False):
        # map almost-one-hot vectors to bin-indices (ints)
        X = super().postprocess_variate(rng, X, batch_mode=True)

        # map bin-indices to real values
        low, high = float(self.space_orig.low), float(self.space_orig.high)
        u = jax.random.uniform(rng, jnp.shape(X))  # u in [0, 1]
        X = low + (X + u) * (high - low) / self.num_bins
        X = jnp.reshape(X, (-1, *self.space_orig.shape))
        return X if batch_mode else X[index]
