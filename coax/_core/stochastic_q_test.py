from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import haiku as hk
from gym.spaces import Discrete, Box

from .._base.test_case import TestCase
from ..utils import safe_sample, quantile_cos_embedding, quantiles, quantiles_uniform
from .stochastic_q import StochasticQ


discrete = Discrete(7)
boxspace = Box(low=0, high=1, shape=(3, 5))
num_bins = 20

Env = namedtuple('Env', ('observation_space', 'action_space'))


def func_type1(S, A, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    logits = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(num_bins),
    ))
    X = jax.vmap(jnp.kron)(S, A)
    return {'logits': logits(X)}


def func_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    logits = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n * num_bins),
        hk.Reshape((discrete.n, num_bins)),
    ))
    return {'logits': logits(S)}


def quantile_net(x, quantile_fractions):
    x_size = x.shape[-1]
    x_tiled = jnp.tile(x[:, None, :], [num_bins, 1])
    quantiles_emb = quantile_cos_embedding(quantile_fractions)
    quantiles_emb = hk.Linear(x_size)(quantiles_emb)
    quantiles_emb = hk.LayerNorm(axis=-1, create_scale=True,
                                 create_offset=True)(quantiles_emb)
    quantiles_emb = jax.nn.sigmoid(quantiles_emb)
    x = x_tiled * quantiles_emb
    x = hk.Linear(x_size)(x)
    x = jax.nn.relu(x)
    return x


def func_quantile_type1(S, A, is_training):
    """ type-1 q-function: (s,a) -> q(s,a) """
    encoder = hk.Sequential((
        hk.Flatten(), hk.Linear(8), jax.nn.relu
    ))
    quantile_fractions = quantiles(batch_size=jax.tree_leaves(S)[0].shape[0],
                                   num_quantiles=num_bins)
    X = jax.vmap(jnp.kron)(S, A)
    x = encoder(X)
    quantile_x = quantile_net(x, quantile_fractions=quantile_fractions)
    quantile_values = hk.Linear(1)(quantile_x)
    return {'values': quantile_values.squeeze(axis=-1),
            'quantile_fractions': quantile_fractions}


def func_quantile_type2(S, is_training):
    """ type-1 q-function: (s,a) -> q(s,a) """
    encoder = hk.Sequential((
        hk.Flatten(), hk.Linear(8), jax.nn.relu
    ))
    quantile_fractions = quantiles_uniform(rng=hk.next_rng_key(),
                                           batch_size=jax.tree_leaves(S)[0].shape[0],
                                           num_quantiles=num_bins)
    x = encoder(S)
    quantile_x = quantile_net(x, quantile_fractions=quantile_fractions)
    quantile_values = hk.Sequential((
        hk.Linear(discrete.n),
        hk.Reshape((discrete.n, num_bins))
    ))(quantile_x)
    return {'values': quantile_values,
            'quantile_fractions': quantile_fractions[:, None, :].tile([1, discrete.n, 1])}


class TestStochasticQ(TestCase):
    def test_init(self):

        # cannot define a type-2 models on a non-discrete action space
        msg = r"type-2 models are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            StochasticQ(func_type2, Env(boxspace, boxspace), (-10, 10), num_bins=num_bins)

        # these should all be fine
        StochasticQ(func_type1, Env(discrete, discrete), (-10, 10), num_bins=num_bins)
        StochasticQ(func_type1, Env(discrete, boxspace), (-10, 10), num_bins=num_bins)
        StochasticQ(func_type1, Env(boxspace, boxspace), (-10, 10), num_bins=num_bins)
        StochasticQ(func_type2, Env(discrete, discrete), (-10, 10), num_bins=num_bins)
        StochasticQ(func_type2, Env(boxspace, discrete), (-10, 10), num_bins=num_bins)
        StochasticQ(func_quantile_type1, Env(discrete, discrete), num_bins=num_bins)
        StochasticQ(func_quantile_type1, Env(discrete, boxspace), num_bins=num_bins)
        StochasticQ(func_quantile_type1, Env(boxspace, boxspace), num_bins=num_bins)
        StochasticQ(func_quantile_type2, Env(discrete, discrete), num_bins=num_bins)
        StochasticQ(func_quantile_type2, Env(boxspace, discrete), num_bins=num_bins)

    # test_call_* ##################################################################################

    def test_call_discrete_discrete_type1(self):
        func = func_type1
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_, logp = q(s, a, return_logp=True)
        print(q_, logp, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for q_ in q(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_call_discrete_discrete_type2(self):
        func = func_type2
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_, logp = q(s, a, return_logp=True)
        print(q_, logp, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for q_ in q(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_call_boxspace_discrete_type1(self):
        func = func_type1
        env = Env(boxspace, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_, logp = q(s, a, return_logp=True)
        print(q_, logp, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for q_ in q(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_call_boxspace_discrete_type2(self):
        func = func_type2
        env = Env(boxspace, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_, logp = q(s, a, return_logp=True)
        print(q_, logp, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for q_ in q(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_call_discrete_boxspace(self):
        func = func_type1
        env = Env(discrete, boxspace)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_, logp = q(s, a, return_logp=True)
        print(q_, logp, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            q(s)

    def test_call_boxspace_boxspace(self):
        func = func_type1
        env = Env(boxspace, boxspace)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_, logp = q(s, a, return_logp=True)
        print(q_, logp, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            q(s)

    # test_mode_* ##################################################################################

    def test_mode_discrete_discrete_type1(self):
        func = func_type1
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_ = q.mode(s, a)
        print(q_, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))

        for q_ in q.mode(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_mode_discrete_discrete_type2(self):
        func = func_type2
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_ = q.mode(s, a)
        print(q_, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))

        for q_ in q.mode(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_mode_boxspace_discrete_type1(self):
        func = func_type1
        env = Env(boxspace, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_ = q.mode(s, a)
        print(q_, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))

        for q_ in q.mode(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_mode_boxspace_discrete_type2(self):
        func = func_type2
        env = Env(boxspace, discrete)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_ = q.mode(s, a)
        print(q_, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))

        for q_ in q.mode(s):
            print(q_, env.observation_space)
            self.assertIn(q_, Box(*value_range, shape=()))

    def test_mode_discrete_boxspace(self):
        func = func_type1
        env = Env(discrete, boxspace)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_ = q.mode(s, a)
        print(q_, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            q.mode(s)

    def test_mode_boxspace_boxspace(self):
        func = func_type1
        env = Env(boxspace, boxspace)
        value_range = (-10, 10)

        s = safe_sample(env.observation_space, seed=17)
        a = safe_sample(env.action_space, seed=18)
        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        q_ = q.mode(s, a)
        print(q_, env.observation_space)
        self.assertIn(q_, Box(*value_range, shape=()))

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            q.mode(s)

    def test_function_state(self):
        func = func_type1
        env = Env(discrete, discrete)
        value_range = (-10, 10)

        q = StochasticQ(func, env, value_range, num_bins=num_bins, random_seed=19)

        print(q.function_state)
        batch_norm_avg = q.function_state['batch_norm/~/mean_ema']['average']
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
            value_range = (-10, 10)
            StochasticQ(badfunc, env, value_range, num_bins=num_bins, random_seed=13)

    def test_bad_output_structure(self):
        def badfunc(S, is_training):
            dist_params = func_type2(S, is_training)
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
            StochasticQ(badfunc, env, value_range, num_bins=num_bins, random_seed=13)
