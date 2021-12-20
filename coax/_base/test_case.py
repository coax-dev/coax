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


__all__ = (
    'DiscreteEnv',
    'BoxEnv',
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
        low=onp.float32(0), high=onp.float32(1), shape=(17, 19))
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
    margin = 0.01  # for robust comparison (x > 0) --> (x > margin)
    decimal = 6    # sets the absolute tolerance

    @property
    def env_discrete(self):
        return DiscreteEnv(self.seed)

    @property
    def env_boxspace(self):
        return BoxEnv(self.seed)

    @property
    def transitions_discrete(self):
        from ..utils import safe_sample
        from ..reward_tracing import TransitionBatch
        return TransitionBatch(
            S=onp.stack([
                safe_sample(self.env_discrete.observation_space, seed=(self.seed * i * 1))
                for i in range(1, 12)], axis=0),
            A=onp.stack([
                safe_sample(self.env_discrete.action_space, seed=(self.seed * i * 2))
                for i in range(1, 12)], axis=0),
            logP=onp.log(onp.random.RandomState(3).rand(11)),
            Rn=onp.random.RandomState(5).randn(11),
            In=onp.random.RandomState(7).randint(2, size=11) * 0.95,
            S_next=onp.stack([
                safe_sample(self.env_discrete.observation_space, seed=(self.seed * i * 11))
                for i in range(1, 12)], axis=0),
            A_next=onp.stack([
                safe_sample(self.env_discrete.action_space, seed=(self.seed * i * 13))
                for i in range(1, 12)], axis=0),
            logP_next=onp.log(onp.random.RandomState(17).rand(11)),
            extra_info=None
        )

    @property
    def transitions_boxspace(self):
        from ..utils import safe_sample
        from ..reward_tracing import TransitionBatch
        return TransitionBatch(
            S=onp.stack([
                safe_sample(self.env_boxspace.observation_space, seed=(self.seed * i * 1))
                for i in range(1, 12)], axis=0),
            A=onp.stack([
                safe_sample(self.env_boxspace.action_space, seed=(self.seed * i * 2))
                for i in range(1, 12)], axis=0),
            logP=onp.log(onp.random.RandomState(3).rand(11)),
            Rn=onp.random.RandomState(5).randn(11),
            In=onp.random.RandomState(7).randint(2, size=11) * 0.95,
            S_next=onp.stack([
                safe_sample(self.env_boxspace.observation_space, seed=(self.seed * i * 11))
                for i in range(1, 12)], axis=0),
            A_next=onp.stack([
                safe_sample(self.env_boxspace.action_space, seed=(self.seed * i * 13))
                for i in range(1, 12)], axis=0),
            logP_next=onp.log(onp.random.RandomState(17).rand(11)),
            extra_info=None
        )

    @property
    def func_pi_discrete(self):
        def func(S, is_training):
            flatten = hk.Flatten()
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)
            seq = hk.Sequential((
                hk.Linear(7),
                batch_norm,
                jnp.tanh,
                hk.Linear(self.env_discrete.action_space.n),
            ))
            return {'logits': seq(flatten(S))}
        return func

    @property
    def func_pi_boxspace(self):
        def func(S, is_training):
            flatten = hk.Flatten()
            batch_norm_m = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm_v = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm_m = partial(batch_norm_m, is_training=is_training)
            batch_norm_v = partial(batch_norm_v, is_training=is_training)
            mu = hk.Sequential((
                hk.Linear(7), batch_norm_m, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(onp.prod(self.env_boxspace.action_space.shape)),
                hk.Reshape(self.env_boxspace.action_space.shape),
            ))
            logvar = hk.Sequential((
                hk.Linear(7), batch_norm_v, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(onp.prod(self.env_boxspace.action_space.shape)),
                hk.Reshape(self.env_boxspace.action_space.shape),
            ))
            return {'mu': mu(flatten(S)), 'logvar': logvar(flatten(S))}
        return func

    @property
    def func_q_type1(self):
        def func(S, A, is_training):
            flatten = hk.Flatten()
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)
            seq = hk.Sequential((
                hk.Linear(7), batch_norm, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(1), jnp.ravel,
            ))
            X = jnp.concatenate((flatten(S), flatten(A)), axis=-1)
            return seq(X)
        return func

    @property
    def func_q_type2(self):
        def func(S, is_training):
            flatten = hk.Flatten()
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)
            seq = hk.Sequential((
                hk.Linear(7), batch_norm, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(self.env_discrete.action_space.n),
            ))
            return seq(flatten(S))
        return func

    @property
    def func_q_stochastic_type1(self):
        def func(S, A, is_training):
            flatten = hk.Flatten()
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)
            seq = hk.Sequential((
                hk.Linear(7),
                batch_norm,
                jnp.tanh,
                hk.Linear(51),
            ))
            print(S.shape, A.shape)
            X = jnp.concatenate((flatten(S), flatten(A)), axis=-1)
            return {'logits': seq(X)}
        return func

    @property
    def func_q_stochastic_type2(self):
        def func(S, is_training):
            flatten = hk.Flatten()
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)
            seq = hk.Sequential((
                hk.Linear(7), batch_norm, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(self.env_discrete.action_space.n * 51),
                hk.Reshape((self.env_discrete.action_space.n, 51))
            ))
            return {'logits': seq(flatten(S))}
        return func

    @property
    def func_v(self):
        def func(S, is_training):
            flatten = hk.Flatten()
            batch_norm = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm = partial(batch_norm, is_training=is_training)
            seq = hk.Sequential((
                hk.Linear(7), batch_norm, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(1), jnp.ravel
            ))
            return seq(flatten(S))
        return func

    @property
    def func_p_type1(self):
        def func(S, A, is_training):
            output_shape = self.env_discrete.observation_space.shape
            flatten = hk.Flatten()
            batch_norm_m = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm_v = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm_m = partial(batch_norm_m, is_training=is_training)
            batch_norm_v = partial(batch_norm_v, is_training=is_training)
            mu = hk.Sequential((
                hk.Linear(7), batch_norm_m, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(onp.prod(output_shape)),
                hk.Reshape(output_shape),
            ))
            logvar = hk.Sequential((
                hk.Linear(7), batch_norm_v, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(onp.prod(output_shape)),
                hk.Reshape(output_shape),
            ))
            X = jnp.concatenate((flatten(S), flatten(A)), axis=-1)
            return {'mu': mu(X), 'logvar': logvar(X)}
        return func

    @property
    def func_p_type2(self):
        def func(S, is_training):
            env = self.env_discrete
            output_shape = (env.action_space.n, *env.observation_space.shape)
            flatten = hk.Flatten()
            batch_norm_m = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm_v = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.95)
            batch_norm_m = partial(batch_norm_m, is_training=is_training)
            batch_norm_v = partial(batch_norm_v, is_training=is_training)
            mu = hk.Sequential((
                hk.Linear(7), batch_norm_m, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(onp.prod(output_shape)),
                hk.Reshape(output_shape),
            ))
            logvar = hk.Sequential((
                hk.Linear(7), batch_norm_v, jnp.tanh,
                hk.Linear(3), jnp.tanh,
                hk.Linear(onp.prod(output_shape)),
                hk.Reshape(output_shape),
            ))
            X = flatten(S)
            return {'mu': mu(X), 'logvar': logvar(X)}
        return func

    def assertArrayAlmostEqual(self, x, y, decimal=None):
        decimal = decimal or self.decimal
        x = onp.asanyarray(x)
        y = onp.asanyarray(y)
        x = (x - x.min()) / (x.max() - x.min() + 1e-16)
        y = (y - y.min()) / (y.max() - y.min() + 1e-16)
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
