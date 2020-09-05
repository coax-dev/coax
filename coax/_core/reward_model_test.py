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
from gym.spaces import Discrete, Box

from .._base.test_case import TestCase
from ..utils import safe_sample
from .reward_model import RewardModel


discrete = Discrete(7)
boxspace = Box(low=0, high=1, shape=(3, 5))


def check_onehot(S):
    if jnp.issubdtype(S.dtype, jnp.integer):
        return hk.one_hot(S, discrete.n)
    return S


def func_type1(S, A, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(1), jnp.ravel,
    ))
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(1), jnp.ravel,
    ))
    X = jax.vmap(jnp.kron)(check_onehot(S), A)
    return {'mu': mu(X), 'logvar': logvar(X)}


def func_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n),
    ))
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n),
    ))
    X = check_onehot(S)
    return {'mu': mu(X), 'logvar': logvar(X)}


class TestRewardModel(TestCase):
    def test_init(self):
        reward_range = (-10, 10)

        # cannot define a type-2 models on a non-discrete action space
        msg = r"type-2 models are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            RewardModel(func_type2, boxspace, boxspace, reward_range)

        # these should all be fine
        RewardModel(func_type1, discrete, discrete, reward_range)
        RewardModel(func_type1, discrete, boxspace, reward_range)
        RewardModel(func_type1, boxspace, boxspace, reward_range)
        RewardModel(func_type2, discrete, discrete, reward_range)
        RewardModel(func_type2, boxspace, discrete, reward_range)

    # test_call_* ##################################################################################

    def test_call_discrete_discrete_type1(self):
        func = func_type1
        observation_space = discrete
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_, logp = r(s, a, return_logp=True)
        print(r_, logp, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for r_ in r(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_call_discrete_discrete_type2(self):
        func = func_type2
        observation_space = discrete
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_, logp = r(s, a, return_logp=True)
        print(r_, logp, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for r_ in r(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_call_boxspace_discrete_type1(self):
        func = func_type1
        observation_space = boxspace
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_, logp = r(s, a, return_logp=True)
        print(r_, logp, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for r_ in r(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_call_boxspace_discrete_type2(self):
        func = func_type2
        observation_space = boxspace
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_, logp = r(s, a, return_logp=True)
        print(r_, logp, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        for r_ in r(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_call_discrete_boxspace(self):
        func = func_type1
        observation_space = discrete
        action_space = boxspace
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_, logp = r(s, a, return_logp=True)
        print(r_, logp, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            r(s)

    def test_call_boxspace_boxspace(self):
        func = func_type1
        observation_space = boxspace
        action_space = boxspace
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_, logp = r(s, a, return_logp=True)
        print(r_, logp, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))
        self.assertArraySubdtypeFloat(logp)
        self.assertArrayShape(logp, ())

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            r(s)

    # test_mode_* ##################################################################################

    def test_mode_discrete_discrete_type1(self):
        func = func_type1
        observation_space = discrete
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_ = r.mode(s, a)
        print(r_, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))

        for r_ in r.mode(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_mode_discrete_discrete_type2(self):
        func = func_type2
        observation_space = discrete
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_ = r.mode(s, a)
        print(r_, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))

        for r_ in r.mode(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_mode_boxspace_discrete_type1(self):
        func = func_type1
        observation_space = boxspace
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_ = r.mode(s, a)
        print(r_, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))

        for r_ in r.mode(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_mode_boxspace_discrete_type2(self):
        func = func_type2
        observation_space = boxspace
        action_space = discrete
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_ = r.mode(s, a)
        print(r_, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))

        for r_ in r.mode(s):
            print(r_, observation_space)
            self.assertIn(r_, Box(*reward_range, shape=()))

    def test_mode_discrete_boxspace(self):
        func = func_type1
        observation_space = discrete
        action_space = boxspace
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_ = r.mode(s, a)
        print(r_, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            r.mode(s)

    def test_mode_boxspace_boxspace(self):
        func = func_type1
        observation_space = boxspace
        action_space = boxspace
        reward_range = (-10, 10)

        s = safe_sample(observation_space, seed=17)
        a = safe_sample(action_space, seed=18)
        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        r_ = r.mode(s, a)
        print(r_, observation_space)
        self.assertIn(r_, Box(*reward_range, shape=()))

        msg = r"input 'A' is required for type-1 dynamics model when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            r.mode(s)

    def test_function_state(self):
        func = func_type1
        observation_space = discrete
        reward_range = (-10, 10)

        action_space = discrete

        r = RewardModel(func, observation_space, action_space, reward_range, random_seed=19)

        print(r.function_state)
        batch_norm_avg = r.function_state['batch_norm/~/mean_ema']['average']
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
            RewardModel(badfunc, boxspace, discrete, (-10, 10), random_seed=13)

    def test_bad_output_structure(self):
        def badfunc(S, is_training):
            dist_params = func_type2(S, is_training)
            dist_params['foo'] = jnp.zeros(1)
            return dist_params
        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\(dict\[\['logvar', 'mu'\]\], \[\*,\*\]\), "
            r"got: PyTreeDef\(dict\[\['foo', 'logvar', 'mu'\]\], \[\*,\*,\*\]\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            RewardModel(badfunc, discrete, discrete, (-10, 10), random_seed=13)
