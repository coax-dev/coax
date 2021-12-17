from functools import partial

import gym
import jax
import haiku as hk
import numpy as onp

from .._base.test_case import TestCase
from .q import Q
from .value_based_policy import EpsilonGreedy, BoltzmannPolicy


env = gym.make('FrozenLakeNonSlippery-v0')


def func_type2(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n),
    ))
    return seq(S)


class TestEpsilonGreedy(TestCase):
    def setUp(self):
        self.env = gym.make('FrozenLakeNonSlippery-v0')
        self.q = Q(func_type2, env)

    def tearDown(self):
        del self.q, self.env

    def test_call(self):
        pi = EpsilonGreedy(self.q, epsilon=0.1)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_greedy(self):
        pi = EpsilonGreedy(self.q, epsilon=0.1)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi.mode(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_dist_params(self):
        pi = EpsilonGreedy(self.q, epsilon=0.1)
        s = self.env.observation_space.sample()
        dist_params = pi.dist_params(s)
        print(onp.exp(dist_params['logits']))
        self.assertEqual(dist_params['logits'].shape, (self.env.action_space.n,))

        # EpsilonGreedy produces logits that are true log-propensities (is not generally the case)
        # this checks if the propensities are properly normalized
        self.assertAlmostEqual(onp.exp(dist_params['logits']).sum(), 1)


class TestBoltzmannPolicy(TestCase):
    def setUp(self):
        self.env = gym.make('FrozenLakeNonSlippery-v0')
        self.q = Q(func_type2, env)

    def tearDown(self):
        del self.q, self.env

    def test_call(self):
        pi = BoltzmannPolicy(self.q, temperature=1.0)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_greedy(self):
        pi = BoltzmannPolicy(self.q, temperature=1.0)
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a = pi.mode(s)
            s, r, done, info = self.env.step(a)
            if done:
                break

    def test_dist_params(self):
        pi = EpsilonGreedy(self.q, epsilon=0.1)
        s = self.env.observation_space.sample()
        dist_params = pi.dist_params(s)
        print(onp.exp(dist_params['logits']))
        self.assertEqual(dist_params['logits'].shape, (self.env.action_space.n,))
