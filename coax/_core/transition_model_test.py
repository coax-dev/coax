from functools import partial
from collections import namedtuple

import gym
import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from .._base.test_case import TestCase
from ..utils import safe_sample
from .transition_model import TransitionModel


discrete = gym.spaces.Discrete(7)
boxspace = gym.spaces.Box(low=0, high=1, shape=(3, 5))

Env = namedtuple('Env', ('observation_space', 'action_space'))


def func_discrete_type1(S, A, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n), jax.nn.softmax
    ))
    X = jax.vmap(jnp.kron)(S, A)
    return seq(X)


def func_discrete_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n * discrete.n),
        hk.Reshape((discrete.n, discrete.n)), jax.nn.softmax
    ))
    return seq(S)


def func_boxspace_type1(S, A, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape)),
        hk.Reshape(boxspace.shape),
    ))
    X = jax.vmap(jnp.kron)(S, A)
    return seq(X)


def func_boxspace_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape) * discrete.n),
        hk.Reshape((discrete.n, *boxspace.shape)),
    ))
    return seq(S)


class TestTransitionModel(TestCase):
    def test_init(self):
        # cannot define a type-2 models on a non-discrete action space
        msg = r"type-2 models are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_boxspace_type2, Env(boxspace, boxspace))
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_discrete_type2, Env(discrete, boxspace))

        msg = r"found leaves with unexpected shapes: \(1(?:, 7)?, 7\) != \(1(?:, 7)?, 3, 5\)"
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_discrete_type1, Env(boxspace, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_discrete_type2, Env(boxspace, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_discrete_type1, Env(boxspace, boxspace))

        msg = r"found leaves with unexpected shapes: \(1(?:, 7)?, 3, 5\) != \(1(?:, 7)?, 7\)"
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_boxspace_type1, Env(discrete, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_boxspace_type2, Env(discrete, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            TransitionModel(func_boxspace_type1, Env(discrete, boxspace))

        # these should all be fine
        TransitionModel(func_discrete_type1, Env(discrete, boxspace))
        TransitionModel(func_discrete_type1, Env(discrete, discrete))
        TransitionModel(func_discrete_type2, Env(discrete, discrete))
        TransitionModel(func_boxspace_type1, Env(boxspace, boxspace))
        TransitionModel(func_boxspace_type1, Env(boxspace, discrete))
        TransitionModel(func_boxspace_type2, Env(boxspace, discrete))

    # test_call_* ##################################################################################

    def test_call_discrete_discrete_type1(self):
        func = func_discrete_type1
        env = Env(discrete, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = TransitionModel(func, env, random_seed=19)

        s_next = p(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_discrete_discrete_type2(self):
        func = func_discrete_type2
        env = Env(discrete, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = TransitionModel(func, env, random_seed=19)

        s_next = p(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_boxspace_discrete_type1(self):
        func = func_boxspace_type1
        env = Env(boxspace, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = TransitionModel(func, env, random_seed=19)

        s_next = p(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_boxspace_discrete_type2(self):
        func = func_boxspace_type2
        env = Env(boxspace, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = TransitionModel(func, env, random_seed=19)

        s_next = p(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_discrete_boxspace(self):
        func = func_discrete_type1
        env = Env(discrete, boxspace)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = TransitionModel(func, env, random_seed=19)

        s_next = p(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            p(s)

    def test_call_boxspace_boxspace(self):
        func = func_boxspace_type1
        env = Env(boxspace, boxspace)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = TransitionModel(func, env, random_seed=19)

        s_next = p(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            p(s)

    # other tests ##################################################################################

    def test_bad_input_signature(self):
        def badfunc(S, is_training, x):
            pass
        msg = (
            r"func has bad signature; "
            r"expected: func\(S, A, is_training\) or func\(S, is_training\), "
            r"got: func\(S, is_training, x\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            env = Env(boxspace, discrete)
            TransitionModel(badfunc, env, random_seed=13)

    def test_bad_output_structure(self):
        def badfunc(S, is_training):
            S_next = func_discrete_type2(S, is_training)
            S_next = (13, S_next)
            return S_next
        msg = (
            r"func has bad return tree_structure, expected: PyTreeDef\(\*\), "
            r"got: PyTreeDef\(\(\*, \*\)\)")
        with self.assertRaisesRegex(TypeError, msg):
            env = Env(discrete, discrete)
            TransitionModel(badfunc, env, random_seed=13)
