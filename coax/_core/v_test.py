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
import haiku as hk

from .._base.test_case import TestCase
from ..utils import get_transition_batch, safe_sample
from .v import V


def func(S, is_training):
    rng1, rng2, rng3 = hk.next_rng_keys(3)
    rate = 0.25 if is_training else 0.
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, rng1, rate),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, rng2, rate),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, rng3, rate),
        partial(batch_norm, is_training=is_training),
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel,
    ))
    return seq(S)


class TestV(TestCase):

    def setUp(self):
        self.v = V(func, self.env_discrete, random_seed=13)
        self.transition_batch = get_transition_batch(self.env_discrete, random_seed=7)

    def tearDown(self):
        del self.v, self.transition_batch

    def test_call(self):
        s = safe_sample(self.env_discrete.observation_space)
        v = self.v(s)
        self.assertAlmostEqual(v, 0.)

    def test_soft_update(self):
        tau = 0.13
        v = self.v
        v_targ = v.copy()
        v.params = jax.tree_map(jnp.ones_like, v.params)
        v_targ.params = jax.tree_map(jnp.zeros_like, v.params)
        expected = jax.tree_map(lambda a: jnp.full_like(a, tau), v.params)
        v_targ.soft_update(v, tau=tau)
        self.assertPytreeAlmostEqual(v_targ.params, expected)

    def test_function_state(self):
        print(self.v.function_state)
        batch_norm_avg = self.v.function_state['batch_norm/~/mean_ema']['average']
        self.assertArrayShape(batch_norm_avg, (1, 8))
        self.assertArrayNotEqual(batch_norm_avg, jnp.zeros_like(batch_norm_avg))

    def test_bad_input_signature(self):
        def badfunc(S, is_training, x):
            pass
        msg = (
            r"func has bad signature; "
            r"expected: func\(S, is_training\), got: func\(S, is_training, x\)")
        with self.assertRaisesRegex(TypeError, msg):
            V(badfunc, self.env_discrete, random_seed=13)

    def test_bad_output_type(self):
        def badfunc(S, is_training):
            return 'garbage'
        msg = r"(?:is not a valid JAX type|func has bad return type)"
        with self.assertRaisesRegex(TypeError, msg):
            V(badfunc, self.env_discrete, random_seed=13)

    def test_bad_output_shape(self):
        def badfunc(S, is_training):
            V = func(S, is_training)
            return jnp.expand_dims(V, axis=-1)
        msg = r"func has bad return shape, expected: \(1,\), got: \(1, 1\)"
        with self.assertRaisesRegex(TypeError, msg):
            V(badfunc, self.env_discrete, random_seed=13)

    def test_bad_output_dtype(self):
        def badfunc(S, is_training):
            V = func(S, is_training)
            return V.astype('int32')
        msg = r"func has bad return dtype; expected a subdtype of jnp\.floating, got dtype=int32"
        with self.assertRaisesRegex(TypeError, msg):
            V(badfunc, self.env_discrete, random_seed=13)
