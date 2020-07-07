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

import numpy as onp
from gym import Wrapper
from gym.spaces import Box
from scipy.special import expit as sigmoid

from .._base.mixins import SpaceUtilsMixin, AddOrigToInfoDictMixin


__all__ = (
    'BoxActionsToReals',
)


class BoxActionsToReals(Wrapper, SpaceUtilsMixin, AddOrigToInfoDictMixin):
    r"""

    This wrapper decompactifies a :class:`Box <gym.spaces.Box>` action space to the reals. This is
    required in order to be able to use a Gaussian policy.

    In practice, the wrapped environment expects the input action
    :math:`a_\text{real}\in\mathbb{R}^n` and then it compactifies it back to a Box of the right
    size:

    .. math::

        a_\text{box}\ =\ \text{low} + (\text{high}-\text{low})\times\text{sigmoid}(a_\text{real})

    Technically, the transformed space is still a Box, but that's only because we assume that the
    values lie between large but finite bounds, :math:`a_\text{real}\in[-10^{15}, 10^{15}]^n`.

    """
    def __init__(self, env):
        super().__init__(env)
        if not self.action_space_is_box:
            raise NotImplementedError("BoxActionsToReals is only implemented for Box action spaces")

        shape_flat = onp.prod(self.env.action_space.shape),
        self.action_space = Box(
            low=onp.full(shape_flat, -1e15, self.env.action_space.dtype),
            high=onp.full(shape_flat, 1e15, self.env.action_space.dtype))

    def _compactify(self, action):
        hi, lo = self.env.action_space.high, self.env.action_space.low
        action = onp.clip(action, -1e15, 1e15)
        action = onp.reshape(action, self.env.action_space.shape)
        return lo + (hi - lo) * sigmoid(action)

    def step(self, a):
        assert self.action_space.contains(a)
        self._a_orig = self._compactify(a)
        s_next, r, done, info = super().step(self._a_orig)
        self._add_a_orig_to_info_dict(info)
        return s_next, r, done, info
