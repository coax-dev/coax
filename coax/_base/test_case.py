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

import gc
import unittest
from collections import namedtuple
from contextlib import AbstractContextManager
from resource import getrusage, RUSAGE_SELF
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from ..wrappers import BoxActionsToReals

__all__ = (
    'DiscreteEnv',
    'BoxEnv',
    'DummyFuncApprox',
    'TestCase',
    'MemoryProfiler',
)


MockEnv = namedtuple('MockEnv', (
    'observation_space', 'action_space', 'reward_range', 'metadata',
    'reset', 'step', 'close', 'random_seed', 'spec'))


def DiscreteEnv(random_seed):
    action_space = gym.spaces.Discrete(3)
    action_space.seed(13 * random_seed)
    observation_space = gym.spaces.Box(
        low=onp.float32(0), high=onp.float32(1), shape=(5,))
    observation_space.seed(11 * random_seed)
    reward_range = (-1., 1.)
    metadata = None
    spec = None

    def reset():
        return observation_space.sample()

    def step(a):
        s = observation_space.sample()
        return s, 0.5, False, {}

    def close():
        pass

    return MockEnv(observation_space, action_space, reward_range, metadata,
                   reset, step, close, random_seed, spec)


def BoxEnv(random_seed):
    env = DiscreteEnv(random_seed)._asdict()
    env['action_space'] = gym.spaces.Box(
        low=onp.float32(0), high=onp.float32(1), shape=(3, 5))
    env['action_space'].seed(7 * random_seed)
    return MockEnv(**env)


def DummyFuncApprox(env, learning_rate=1.):
    from .._core.func_approx import FuncApprox  # avoid circular dependence

    class cls(FuncApprox):
        def body(self, S, is_training):
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)

            seq = hk.Sequential((
                hk.Linear(7), batch_norm, jnp.tanh,
                hk.Linear(1), jax.nn.sigmoid,
            ))
            return seq(S)

    return cls(env, random_seed=(17 * env.random_seed), learning_rate=learning_rate)


class MemoryProfiler(AbstractContextManager):
    def __enter__(self):
        gc.collect()
        self._mem_used = None
        self._mem_start = self._get_mem()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        self._mem_used = self._get_mem() - self._mem_start

    @property
    def memory_used(self):
        gc.collect()
        if self._mem_used is not None:
            return self._mem_used  # only available after __exit__
        else:
            return self._get_mem() - self._mem_start

    @staticmethod
    def _get_mem():
        return getrusage(RUSAGE_SELF).ru_maxrss


class TestCase(unittest.TestCase):
    r""" adds some common properties to unittest.TestCase """
    seed = 42
    margin = 0.1  # for robust comparison (x > 0) --> (x > margin)
    decimal = 6

    @property
    def env_discrete(self):
        return DiscreteEnv(self.seed)

    @property
    def env_box(self):
        return BoxEnv(self.seed)

    @property
    def env_box_decompactified(self):
        return BoxActionsToReals(self.env_box)

    def assertArrayAlmostEqual(self, x, y, decimal=None):
        decimal = decimal or self.decimal
        onp.testing.assert_array_almost_equal(x, y, decimal=decimal)

    def assertArrayNotEqual(self, x, y, margin=margin):
        reldiff = jnp.abs(2 * (x - y) / (x + y + 1e-16))
        maxdiff = jnp.max(reldiff)
        assert float(maxdiff) > self.margin

    def assertPytreeAlmostEqual(self, x, y, decimal=None):
        decimal = decimal or self.decimal
        jax.tree_multimap(
            lambda x, y: onp.testing.assert_array_almost_equal(
                x, y, decimal=decimal), x, y)

    def assertPytreeNotEqual(self, x, y, margin=None):
        margin = margin or self.margin
        reldiff = jax.tree_multimap(
            lambda a, b: abs(2 * (a - b) / (a + b + 1e-16)), x, y)
        maxdiff = max(jnp.max(d) for d in jax.tree_leaves(reldiff))
        assert float(maxdiff) > margin

    def assertArraySubdtypeFloat(self, arr):
        self.assertTrue(jnp.issubdtype(arr.dtype, jnp.floating))

    def assertArraySubdtypeInt(self, arr):
        self.assertTrue(jnp.issubdtype(arr.dtype, jnp.integer))

    def assertArrayShape(self, arr, shape):
        self.assertEqual(arr.shape, shape)

    def assertAlmostEqual(self, x, y, decimal=None):
        decimal = decimal or self.decimal
        super().assertAlmostEqual(x, y, places=decimal)
