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

import gym
import jax.numpy as jnp
import numpy as onp

from ..utils import docstring
from .policy import Policy


__all__ = (
    'RandomPolicy',
)


class RandomPolicy:
    r"""

    A simple random policy.

    Parameters
    ----------
    env : gym.Env

        The gym-style environment. This is only used to get the :code:`env.action_space`.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, env, random_seed=None):
        if not isinstance(env.action_space, gym.Space):
            raise TypeError(f"env.action_space must be a gym.Space, got: {type(env.action_space)}")
        self.action_space = env.action_space
        self.action_space.seed(random_seed)
        self.random_seed = random_seed

    @docstring(Policy.__call__)
    def __call__(self, s, return_logp=False):
        a = self.action_space.sample()
        if not return_logp:
            return a

        if isinstance(self.action_space, gym.spaces.Discrete):
            logp = -onp.log(self.num_actions)
            return a, logp

        if isinstance(self.action_space, gym.spaces.Box):
            sizes = self.action_space.high - self.action_space.low
            logp = -onp.sum(onp.log(sizes))  # log(prod(1/sizes))
            return a, logp

        raise NotImplementedError(
            "the log-propensity of a 'uniform' distribution over a "
            f"{self.action_space.__class__.__name__} space is not yet implemented; "
            "please submit a feature request")

    @docstring(Policy.mode)
    def mode(self, s):
        return self(s, return_logp=False)

    @docstring(Policy.dist_params)
    def dist_params(self, s):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return {'logits': jnp.zeros(self.action_space.n)}

        if isinstance(self.action_space, gym.spaces.Box):
            return {
                'mu': jnp.zeros(self.action_space.shape),
                'logvar': 15 * jnp.ones(self.action_space.shape)}

        raise NotImplementedError(
            "the dist_params of a 'uniform' distribution over a "
            f"{self.action_space.__class__.__name__} space is not yet implemented; "
            "please submit a feature request")
