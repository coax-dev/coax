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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from coax._core.quantile_q import QuantileQ

from .._base.test_case import TestCase, DiscreteEnv, BoxEnv
from ..utils import safe_sample


env_discrete = DiscreteEnv(random_seed=13)
env_boxspace = BoxEnv(random_seed=17)
num_quantiles = 5
quantile_embedding_dim = 4


def quantile_network(x, quantiles):
    state_vector_length = x.shape[-1]
    quantile_net = jnp.tile(quantiles[..., None], [1, quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = hk.Linear(state_vector_length)(quantile_net)
    quantile_net = jax.nn.relu(quantile_net)
    x = x[:, None, ...] * quantile_net
    x = hk.Linear(state_vector_length)(x)
    x = jax.nn.relu(x)
    return x


def func_type3(S, A, quantiles, is_training):
    encoder_network = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
    ))
    S = hk.Flatten()(S)
    A = hk.Flatten()(A)
    X = jnp.concatenate((S, A), axis=-1)
    x = encoder_network(X)
    x = quantile_network(x, quantiles)
    x = hk.Linear(1)(x)
    return x.reshape(-1, num_quantiles)


def func_type4(S, quantiles, is_training):
    encoder_network = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
    ))
    x = encoder_network(S)
    x = quantile_network(x, quantiles)
    x = hk.Linear(env_discrete.action_space.n)(x)
    x = jnp.moveaxis(x, -1, -2)
    return x


class TestQuantileQ(TestCase):
    decimal = 5

    def test_init(self):
        # cannot define a type-4 q-function on a non-discrete action space
        msg = r"type-4 q-functions are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            QuantileQ(func_type4, env_boxspace, num_quantiles)

        # these should all be fine
        QuantileQ(func_type3, env_boxspace, num_quantiles)
        QuantileQ(func_type3, env_discrete, num_quantiles)
        QuantileQ(func_type4, env_discrete, num_quantiles)

    def test_call_type3_discrete(self):
        env = env_discrete
        func = func_type3
        s = safe_sample(env.observation_space, seed=19)
        a = safe_sample(env.action_space, seed=19)
        q = QuantileQ(func, env, num_quantiles, random_seed=42)

        # without a
        q_s = q(s)
        self.assertArrayShape(q_s, (env.action_space.n, ))
        self.assertArraySubdtypeFloat(q_s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_s)
        self.assertArrayAlmostEqual(q_sa, q_s[a])

    def test_call_type4_discrete(self):
        env = env_discrete
        func = func_type4
        s = safe_sample(env.observation_space, seed=19)
        a = safe_sample(env.action_space, seed=19)
        q = QuantileQ(func, env, num_quantiles, random_seed=42)

        # without a
        q_s = q(s)
        self.assertArrayShape(q_s, (env.action_space.n,))
        self.assertArraySubdtypeFloat(q_s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_s)
        self.assertArrayAlmostEqual(q_sa, q_s[a])

    def test_call_type3_box(self):
        env = env_boxspace
        func = func_type3
        s = safe_sample(env.observation_space, seed=19)
        a = safe_sample(env.action_space, seed=19)
        q = QuantileQ(func, env, num_quantiles, random_seed=42)

        # type-1 requires a if actions space is non-discrete
        msg = r"input 'A' is required for type-3 q-function when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            q(s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_sa)

    def test_apply_q3_as_q4(self):
        env = env_discrete
        func = func_type3
        q = QuantileQ(func, env, num_quantiles, random_seed=42)
        n = env.action_space.n  # num_actions

        def q3_func(params, state, rng, S, A, is_training):
            A = jnp.argmax(A, axis=1)
            return jnp.array([encode(s, a) for s, a in zip(S, A)]), state

        def encode(s, a):
            return 2 ** a + 2 ** (s + n)

        def decode(i):
            b = onp.array(list(bin(i)))[::-1]
            a = onp.argwhere(b[:n] == '1').item()
            s = onp.argwhere(b[n:] == '1').item()
            return s, a

        q._function = q3_func
        rng = jax.random.PRNGKey(0)
        params = ()
        state = ()
        is_training = True

        S = jnp.array([5, 7, 11, 13, 17, 19, 23])
        encoded_rows, _ = q.function_type2(params, state, rng, S, is_training)
        for s, encoded_row in zip(S, encoded_rows):
            for a, x in enumerate(encoded_row):
                s_, a_ = decode(x)
                self.assertEqual(s_, s)
                self.assertEqual(a_, a)

    def test_apply_q4_as_q3(self):
        env = env_discrete
        func = func_type4
        q = QuantileQ(func, env, num_quantiles, random_seed=42)
        n = env.action_space.n  # num_actions

        def q4_func(params, state, rng, S, is_training):
            batch_size = jax.tree_leaves(S)[0].shape[0]
            return jnp.tile(jnp.arange(n), reps=(batch_size, 1)), state

        q._function = q4_func
        rng = jax.random.PRNGKey(0)
        params = ()
        state = ()
        is_training = True

        S = jnp.array([5, 7, 11, 13, 17, 19, 23])
        A = jnp.array([2, 0, 1, 1, 0, 1, 2])
        A_onehot = q.action_preprocessor(q.rng, A)
        Q_sa, _ = q.function_type1(params, state, rng, S, A_onehot, is_training)
        self.assertArrayAlmostEqual(Q_sa, A)

    def test_soft_update(self):
        tau = 0.13
        env = env_discrete
        func = func_type3
        q = QuantileQ(func, env, num_quantiles, random_seed=42)
        q_targ = q.copy()
        q.params = jax.tree_map(jnp.ones_like, q.params)
        q_targ.params = jax.tree_map(jnp.zeros_like, q.params)
        expected = jax.tree_map(lambda a: jnp.full_like(a, tau), q.params)
        q_targ.soft_update(q, tau=tau)
        self.assertPytreeAlmostEqual(q_targ.params, expected)

    def test_function_state(self):
        env = env_discrete
        func = func_type3
        q = QuantileQ(func, env, num_quantiles, random_seed=42)
        print(q.function_state)
        batch_norm_avg = q.function_state['batch_norm/~/mean_ema']['average']
        self.assertArrayShape(batch_norm_avg, (1, 8))
        self.assertArrayNotEqual(batch_norm_avg, jnp.zeros_like(batch_norm_avg))

    def test_bad_input_signature(self):
        env = env_discrete

        def badfunc(S, A, is_training, x):
            pass
        msg = (
            r"func has bad signature; "
            r"expected: func\(S, A, quantiles, is_training\) or func\(S, quantiles, is_training\), "
            r"got: func\(S, A, is_training, x\)")
        with self.assertRaisesRegex(TypeError, msg):
            QuantileQ(badfunc, env, num_quantiles)

    def test_bad_output_type(self):
        env = env_discrete

        def badfunc(S, A, quantiles, is_training):
            return 'garbage'
        msg = r"(?:is not a valid JAX type|func has bad return type)"
        with self.assertRaisesRegex(TypeError, msg):
            QuantileQ(badfunc, env, num_quantiles)

    def test_bad_output_shape_type3(self):
        env = env_discrete

        def badfunc(S, A, quantiles, is_training):
            Q = func_type3(S, A, quantiles, is_training)
            return jnp.expand_dims(Q, axis=-1)
        msg = r"func has bad return shape, expected: \(1, 5\), got: \(1, 5, 1\)"
        with self.assertRaisesRegex(TypeError, msg):
            QuantileQ(badfunc, env, num_quantiles)

    def test_bad_output_shape_type4(self):
        env = env_discrete

        def badfunc(S, quantiles, is_training):
            Q = func_type4(S, quantiles, is_training)
            return Q[:, :2]
        msg = r"func has bad return shape, expected: \(1, 3, 5\), got: \(1, 2, 5\)"
        with self.assertRaisesRegex(TypeError, msg):
            QuantileQ(badfunc, env, num_quantiles)

    def test_bad_output_dtype(self):
        env = env_discrete

        def badfunc(S, A, quantiles, is_training):
            Q = func_type3(S, A, quantiles, is_training)
            return Q.astype('int32')
        msg = r"func has bad return dtype; expected a subdtype of jnp\.floating, got dtype=int32"
        with self.assertRaisesRegex(TypeError, msg):
            QuantileQ(badfunc, env, num_quantiles)