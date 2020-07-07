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

from .._base.mixins import PolicyMixin
from ..utils import docstring


__all__ = (
    'RandomPolicy',
)


class RandomPolicy:
    r"""

    A simple random policy.

    Parameters
    ----------
    env : gym environment

        A gym-style environment.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, env, random_seed=None):
        self.env = env
        self.random_seed = random_seed
        self.env.action_space.seed(random_seed)

    @docstring(PolicyMixin.__call__)
    def __call__(self, s, return_logp=False):
        if return_logp:
            if self.action_space_is_discrete:
                logp = -onp.log(self.num_actions)
            elif self.action_space_is_box:
                sizes = self.env.action_space.high - self.env.action_space.low
                logp = -onp.sum(onp.log(sizes))  # log(prod(1/sizes))
            else:
                raise NotImplementedError(
                    "the log-propensity of a 'uniform' distribution over a "
                    f"{self.env.action_space} is not yet implemented; "
                    "please submit a feature request")
        a = self.env.action_space.sample()
        return (a, logp) if return_logp else a

    __call__.__doc__ = PolicyMixin.__call__.__doc__
