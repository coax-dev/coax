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

from ..utils import safe_sample, single_to_batch
from ..proba_dists import ProbaDist
from .base_func import BaseFunc, ExampleData, Inputs, ArgsType2
from .base_policy import PolicyMixin


class Policy(BaseFunc, PolicyMixin):
    r"""

    A parametrized policy :math:`\pi_\theta(a|s)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations that are compatible with the
        corresponding probability distribution. If left unspecified, this defaults to:

        .. code:: python

            observation_preprocessor = ProbaDist(observation_space).preprocess_variate

        See also :attr:`coax.proba_dists.ProbaDist.preprocess_variate`.

    proba_dist : ProbaDist, optional

        A probability distribution that is used to interpret the output of :code:`func
        <coax.Policy.func>`. Check out the :mod:`coax.proba_dists` module for available options.

        If left unspecified, this defaults to:

        .. code:: python

            proba_dist = coax.proba_dists.ProbaDist(action_space)

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(self, func, env, observation_preprocessor=None, proba_dist=None, random_seed=None):
        self.observation_preprocessor = observation_preprocessor
        self.proba_dist = proba_dist

        # defaults
        if self.observation_preprocessor is None:
            self.observation_preprocessor = ProbaDist(env.observation_space).preprocess_variate
        if self.proba_dist is None:
            self.proba_dist = ProbaDist(env.action_space)

        super().__init__(
            func=func,
            observation_space=env.observation_space,
            action_space=env.action_space,
            random_seed=random_seed)

    @classmethod
    def example_data(
            cls, env, observation_preprocessor=None, proba_dist=None,
            batch_size=1, random_seed=None):

        if not isinstance(env.observation_space, Space):
            raise TypeError(
                "env.observation_space must be derived from gym.Space, "
                f"got: {type(env.observation_space)}")
        if not isinstance(env.action_space, Space):
            raise TypeError(
                f"env.action_space must be derived from gym.Space, got: {type(env.action_space)}")

        if observation_preprocessor is None:
            observation_preprocessor = single_to_batch

        rnd = onp.random.RandomState(random_seed)

        # input: state observations
        S = [safe_sample(env.observation_space, rnd) for _ in range(batch_size)]
        S = [observation_preprocessor(s) for s in S]
        S = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *S)

        # output
        if proba_dist is None:
            proba_dist = ProbaDist(env.action_space)
        dist_params = jax.tree_map(
            lambda x: jnp.asarray(rnd.randn(batch_size, *x.shape[1:])), proba_dist.default_priors)

        return ExampleData(
            inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
            output=dist_params,
        )

    def _check_signature(self, func):
        if tuple(signature(func).parameters) != ('S', 'is_training'):
            sig = ', '.join(signature(func).parameters)
            raise TypeError(
                f"func has bad signature; expected: func(S, is_training), got: func({sig})")

        Env = namedtuple('Env', ('observation_space', 'action_space'))
        return self.example_data(
            env=Env(self.observation_space, self.action_space),
            observation_preprocessor=self.observation_preprocessor,
            proba_dist=self.proba_dist,
            batch_size=1,
            random_seed=self.random_seed,
        )

    def _check_output(self, actual, expected):
        expected_leaves, expected_structure = jax.tree_flatten(expected)
        actual_leaves, actual_structure = jax.tree_flatten(actual)
        assert all(isinstance(x, jnp.ndarray) for x in expected_leaves), "bad example_data"

        if actual_structure != expected_structure:
            raise TypeError(
                f"func has bad return tree_structure, expected: {expected_structure}, "
                f"got: {actual_structure}")

        if not all(isinstance(x, jnp.ndarray) for x in actual_leaves):
            bad_types = tuple(type(x) for x in actual_leaves if not isinstance(x, jnp.ndarray))
            raise TypeError(
                "all leaves of dist_params must be of type: jax.numpy.ndarray, "
                f"found leaves of type: {bad_types}")

        if not all(a.shape == b.shape for a, b in zip(actual_leaves, expected_leaves)):
            shapes_tree = jax.tree_multimap(
                lambda a, b: f"{a.shape} {'!=' if a.shape != b.shape else '=='} {b.shape}",
                actual, expected)
            raise TypeError(f"found leaves with unexpected shapes: {shapes_tree}")
