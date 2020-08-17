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

from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
from gym.spaces import Space

from ..utils import single_to_batch, safe_sample
from .base_func import BaseFunc, ExampleData, Inputs


__all__ = (
    'V',
)


Args = namedtuple('Args', ('S', 'is_training'))


class V(BaseFunc):
    r"""

    A state value function :math:`v_\theta(s)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass. The function signature must be the
        same as the example below.

    observation_space : gym.Space

        The observation space of the environment. This is used to generate example input for
        initializing ``func``. This is done after Haiku-transforming it, see also
        :func:`haiku.transform_with_state`.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(self, func, observation_space, random_seed=None):
        super().__init__(
            func=func,
            observation_space=observation_space,
            action_space=None,
            random_seed=random_seed)

    def __call__(self, s):
        r"""

        Evaluate the value function on a state observation :math:`s`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        v : ndarray, shape: ()

            The estimated expected value associated with the input state observation ``s``.

        """
        S = single_to_batch(s)
        V, _ = self.function(self.params, self.function_state, self.rng, S, False)
        return onp.asarray(V[0])

    @classmethod
    def example_data(cls, observation_space, batch_size=1, random_seed=None):

        if not isinstance(observation_space, Space):
            raise TypeError(
                f"observation_space must be derived from gym.Space, got: {type(observation_space)}")

        rnd = onp.random.RandomState(random_seed)

        # input: state observations
        S = [safe_sample(observation_space, rnd) for _ in range(batch_size)]
        S = jax.tree_multimap(lambda *x: jnp.stack(x, axis=0), *S)

        return ExampleData(
            inputs=Inputs(args=Args(S=S, is_training=True), static_argnums=(1,)),
            output=jnp.asarray(rnd.randn(batch_size)),
        )

    def _check_signature(self, func):
        if tuple(signature(func).parameters) != ('S', 'is_training'):
            sig = ', '.join(signature(func).parameters)
            raise TypeError(
                f"func has bad signature; expected: func(S, is_training), got: func({sig})")

        # example inputs
        return self.example_data(
            observation_space=self.observation_space,
            batch_size=1,
            random_seed=self.random_seed,
        )

    def _check_output(self, actual, expected):
        if not isinstance(actual, jnp.ndarray):
            raise TypeError(
                f"func has bad return type; expected jnp.ndarray, got {actual.__class__.__name__}")

        if not jnp.issubdtype(actual.dtype, jnp.floating):
            raise TypeError(
                "func has bad return dtype; expected a subdtype of jnp.floating, "
                f"got dtype={actual.dtype}")

        if actual.shape != expected.shape:
            raise TypeError(
                f"func has bad return shape, expected: {expected.shape}, got: {actual.shape}")
