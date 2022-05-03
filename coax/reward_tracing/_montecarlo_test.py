from itertools import islice

import pytest
import gym
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

from .._base.errors import InsufficientCacheError
from ..utils import check_array
from ._montecarlo import MonteCarlo


class MockEnv:
    action_space = gym.spaces.Discrete(10)


class TestMonteCarlo:
    env = MockEnv()
    gamma = 0.85
    S = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    A = jnp.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7])
    # P = jnp.array([
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # a=6
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # a=3
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # a=7
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # a=4
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # a=6
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # a=9
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # a=2
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # a=6
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # a=7
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # a=4
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # a=3
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # a=7
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # a=7
    # ])
    R = jnp.array(
        [-0.48, 0.16, 0.23, 0.11, 1.46, 1.53, -2.43, 0.60, -0.25, -0.16, -1.47, 1.48, -0.02])
    D = jnp.array([False] * 12 + [True])
    G = jnp.zeros_like(R)
    for i, r in enumerate(R[::-1]):
        G = G.at[i].set(r + gamma * G[i - 1])
    G = G[::-1]
    episode = list(zip(S, A, R, D))

    def test_append_pop_too_soon(self):
        cache = MonteCarlo(self.gamma)
        for s, a, r, done in self.episode:
            cache.add(s, a, r, done)
            break

        with pytest.raises(InsufficientCacheError):
            cache.pop()

    def test_append_pop_expected(self):
        cache = MonteCarlo(self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i

        assert cache
        assert len(cache) == 13

        for i in range(13):
            assert cache
            transition = cache.pop()
            check_array(transition.S, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.A, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.Rn, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.In, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.S_next, ndim=1, axis_size=1, axis=0, except_np=True)
            assert_array_almost_equal(transition.S[0], self.S[12 - i])
            assert_array_almost_equal(transition.A[0], self.A[12 - i])
            assert_array_almost_equal(transition.Rn[0], self.G[12 - i])
            assert_array_almost_equal(transition.In[0], jnp.zeros(0))

        assert not cache

    def test_append_flush_too_soon(self):
        cache = MonteCarlo(self.gamma)
        for i, (s, a, r, done) in islice(enumerate(self.episode, 1), 4):
            cache.add(s, a, r, done)
            assert len(cache) == i

        with pytest.raises(InsufficientCacheError):
            cache.flush()

    def test_append_flush_expected(self):
        cache = MonteCarlo(self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i

        transitions = cache.flush()
        assert_array_almost_equal(transitions.S, self.S[::-1])
        assert_array_almost_equal(transitions.A, self.A[::-1])
        assert_array_almost_equal(transitions.Rn, self.G[::-1])
        assert_array_almost_equal(transitions.In, jnp.zeros(i))
