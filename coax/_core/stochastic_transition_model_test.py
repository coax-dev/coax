from functools import partial
from collections import namedtuple

import gym
import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from .._base.test_case import TestCase
from ..utils import safe_sample
from .stochastic_transition_model import StochasticTransitionModel


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
        hk.Linear(discrete.n),
    ))
    X = jax.vmap(jnp.kron)(S, A)
    return {'logits': seq(X)}


def func_discrete_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n * discrete.n),
        hk.Reshape((discrete.n, discrete.n)),
    ))
    return {'logits': seq(S)}


def func_boxspace_type1(S, A, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape)),
        hk.Reshape(boxspace.shape),
    ))
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape)),
        hk.Reshape(boxspace.shape),
    ))
    X = jax.vmap(jnp.kron)(S, A)
    return {'mu': mu(X), 'logvar': logvar(X)}


def func_boxspace_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape) * discrete.n),
        hk.Reshape((discrete.n, *boxspace.shape)),
    ))
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape) * discrete.n),
        hk.Reshape((discrete.n, *boxspace.shape)),
    ))
    return {'mu': mu(S), 'logvar': logvar(S)}


class TestStochasticTransitionModel(TestCase):
    def test_init(self):
        # cannot define a type-2 models on a non-discrete action space
        msg = r"type-2 models are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_boxspace_type2, Env(boxspace, boxspace))
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_discrete_type2, Env(discrete, boxspace))

        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\({'logvar': \*, 'mu': \*}\), "
            r"got: PyTreeDef\({'logits': \*}\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_discrete_type1, Env(boxspace, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_discrete_type2, Env(boxspace, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_discrete_type1, Env(boxspace, boxspace))

        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\({'logits': \*}\), "
            r"got: PyTreeDef\({'logvar': \*, 'mu': \*}\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_boxspace_type1, Env(discrete, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_boxspace_type2, Env(discrete, discrete))
        with self.assertRaisesRegex(TypeError, msg):
            StochasticTransitionModel(func_boxspace_type1, Env(discrete, boxspace))

        # these should all be fine
        StochasticTransitionModel(func_discrete_type1, Env(discrete, boxspace))
        StochasticTransitionModel(func_discrete_type1, Env(discrete, discrete))
        StochasticTransitionModel(func_discrete_type2, Env(discrete, discrete))
        StochasticTransitionModel(func_boxspace_type1, Env(boxspace, boxspace))
        StochasticTransitionModel(func_boxspace_type1, Env(boxspace, discrete))
        StochasticTransitionModel(func_boxspace_type2, Env(boxspace, discrete))

    # test_call_* ##################################################################################

    def test_call_discrete_discrete_type1(self):
        func = func_discrete_type1
        env = Env(discrete, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next, logp = p(s, a, return_logp=True)
        print(s_next, logp, env.observation_space)
        self.assertIn(s_next, env.observation_space)
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_discrete_discrete_type2(self):
        func = func_discrete_type2
        env = Env(discrete, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next, logp = p(s, a, return_logp=True)
        print(s_next, logp, env.observation_space)
        self.assertIn(s_next, env.observation_space)
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_boxspace_discrete_type1(self):
        func = func_boxspace_type1
        env = Env(boxspace, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next, logp = p(s, a, return_logp=True)
        print(s_next, logp, env.observation_space)
        self.assertIn(s_next, env.observation_space)
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_boxspace_discrete_type2(self):
        func = func_boxspace_type2
        env = Env(boxspace, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next, logp = p(s, a, return_logp=True)
        print(s_next, logp, env.observation_space)
        self.assertIn(s_next, env.observation_space)
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for s_next in p(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_call_discrete_boxspace(self):
        func = func_discrete_type1
        env = Env(discrete, boxspace)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next, logp = p(s, a, return_logp=True)
        print(s_next, logp, env.observation_space)
        self.assertIn(s_next, env.observation_space)
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            p(s)

    def test_call_boxspace_boxspace(self):
        func = func_boxspace_type1
        env = Env(boxspace, boxspace)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next, logp = p(s, a, return_logp=True)
        print(s_next, logp, env.observation_space)
        self.assertIn(s_next, env.observation_space)
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            p(s)

    # test_mode_* ##################################################################################

    def test_mode_discrete_discrete_type1(self):
        func = func_discrete_type1
        env = Env(discrete, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next = p.mode(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p.mode(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_mode_discrete_discrete_type2(self):
        func = func_discrete_type2
        env = Env(discrete, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next = p.mode(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p.mode(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_mode_boxspace_discrete_type1(self):
        func = func_boxspace_type1
        env = Env(boxspace, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next = p.mode(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p.mode(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_mode_boxspace_discrete_type2(self):
        func = func_boxspace_type2
        env = Env(boxspace, discrete)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next = p.mode(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        for s_next in p.mode(s):
            print(s_next, env.observation_space)
            self.assertIn(s_next, env.observation_space)

    def test_mode_discrete_boxspace(self):
        func = func_discrete_type1
        env = Env(discrete, boxspace)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next = p.mode(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            p.mode(s)

    def test_mode_boxspace_boxspace(self):
        func = func_boxspace_type1
        env = Env(boxspace, boxspace)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        p = StochasticTransitionModel(func, env, random_seed=19)

        s_next = p.mode(s, a)
        print(s_next, env.observation_space)
        self.assertIn(s_next, env.observation_space)

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            p.mode(s)

    def test_function_state(self):
        func = func_discrete_type1
        env = Env(discrete, discrete)

        p = StochasticTransitionModel(func, env, random_seed=19)

        print(p.function_state)
        batch_norm_avg = p.function_state['batch_norm/~/mean_ema']['average']
        self.assertArrayShape(batch_norm_avg, (1, 8))
        self.assertArrayNotEqual(batch_norm_avg, jnp.zeros_like(batch_norm_avg))

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
            StochasticTransitionModel(badfunc, env, random_seed=13)

    def test_bad_output_structure(self):
        def badfunc(S, is_training):
            dist_params = func_discrete_type2(S, is_training)
            dist_params['foo'] = jnp.zeros(1)
            return dist_params
        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\({'logits': \*}\), "
            r"got: PyTreeDef\({'foo': \*, 'logits': \*}\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            env = Env(discrete, discrete)
            StochasticTransitionModel(badfunc, env, random_seed=13)
