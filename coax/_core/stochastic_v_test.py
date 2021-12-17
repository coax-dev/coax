from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import haiku as hk
from gym.spaces import Discrete, Box

from .._base.test_case import TestCase
from ..utils import safe_sample
from .stochastic_v import StochasticV

discrete = Discrete(7)
boxspace = Box(low=0, high=1, shape=(3, 5))
num_bins = 20

Env = namedtuple('Env', ('observation_space', 'action_space'))


def func(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    logits = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(num_bins),
    ))
    return {'logits': logits(S)}


class TestStochasticV(TestCase):
    def test_init(self):
        StochasticV(func, Env(boxspace, boxspace), (-10, 10), num_bins=num_bins)
        StochasticV(func, Env(boxspace, discrete), (-10, 10), num_bins=num_bins)
        StochasticV(func, Env(discrete, boxspace), (-10, 10), num_bins=num_bins)
        StochasticV(func, Env(discrete, discrete), (-10, 10), num_bins=num_bins)

    # test_call_* ##################################################################################

    def test_call_discrete(self):
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        v = StochasticV(func, env, value_range, num_bins=num_bins, random_seed=19)

        v_, logp = v(s, return_logp=True)
        print(v_, logp, env.observation_space)
        self.assertIn(v_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

    def test_call_boxspace(self):
        env = Env(boxspace, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        v = StochasticV(func, env, value_range, num_bins=num_bins, random_seed=19)

        v_, logp = v(s, return_logp=True)
        print(v_, logp, env.observation_space)
        self.assertIn(v_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

    # test_mode_* ##################################################################################

    def test_mode_discrete(self):
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        v = StochasticV(func, env, value_range, num_bins=num_bins, random_seed=19)

        v_ = v.mode(s)
        print(v_, env.observation_space)
        self.assertIn(v_, Box(*value_range, shape=()))

    def test_mode_boxspace(self):
        env = Env(boxspace, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        v = StochasticV(func, env, value_range, num_bins=num_bins, random_seed=19)

        v_ = v.mode(s)
        print(v_, env.observation_space)
        self.assertIn(v_, Box(*value_range, shape=()))

    def test_function_state(self):
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        v = StochasticV(func, env, value_range, num_bins=num_bins, random_seed=19)

        print(v.function_state)
        batch_norm_avg = v.function_state['batch_norm/~/mean_ema']['average']
        self.assertArrayShape(batch_norm_avg, (1, 8))
        self.assertArrayNotEqual(batch_norm_avg, jnp.zeros_like(batch_norm_avg))

    # other tests ##################################################################################

    def test_bad_input_signature(self):
        def badfunc(S, is_training, x):
            pass
        msg = (
            r"func has bad signature; "
            r"expected: func\(S, is_training\), "
            r"got: func\(S, is_training, x\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            env = Env(boxspace, discrete)
            value_range = (-10, 10)
            StochasticV(badfunc, env, value_range, num_bins=num_bins, random_seed=13)

    def test_bad_output_structure(self):
        def badfunc(S, is_training):
            dist_params = func(S, is_training)
            dist_params['foo'] = jnp.zeros(1)
            return dist_params
        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\({'logits': \*}\), "
            r"got: PyTreeDef\({'foo': \*, 'logits': \*}\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            env = Env(discrete, discrete)
            value_range = (-10, 10)
            StochasticV(badfunc, env, value_range, num_bins=num_bins, random_seed=13)
