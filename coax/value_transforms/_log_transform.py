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

import jax.numpy as jnp

from ._base import ValueTransform


class LogTransform(ValueTransform):
    r"""

    A simple invertible log-transform.


    .. math::

        x\ \mapsto\ y\ =\ \lambda\,\text{sign}(x)\,
            \log\left(1+\frac{|x|}{\lambda}\right)

    with inverse:

    .. math::

        y\ \mapsto\ x\ =\ \lambda\,\text{sign}(y)\,
            \left(\text{e}^{|y|/\lambda} - 1\right)

    This transform logarithmically supresses large values :math:`|x|\gg1` and smoothly interpolates
    to the identity transform for small values :math:`|x|\sim1` (see figure below).

    .. image:: /_static/img/log_transform.svg
        :alt: Invertible log-transform
        :width: 640px

    Parameters
    ----------
    scale : positive float, optional

        The scale :math:`\lambda>0` of the linear-to-log cross-over. Smaller
        values for :math:`\lambda` translate into earlier onset of the
        cross-over.

    """
    __slots__ = ValueTransform.__slots__ + ('scale',)

    def __init__(self, scale=1.0):
        assert scale > 0
        self.scale = scale

        def transform_func(x):
            return jnp.sign(x) * scale * jnp.log(1 + jnp.abs(x) / scale)

        def inverse_func(x):
            return jnp.sign(x) * scale * (jnp.exp(jnp.abs(x) / scale) - 1)

        self._transform_func = transform_func
        self._inverse_func = inverse_func
