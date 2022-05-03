from itertools import islice

import pytest
import gym
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

from .._base.errors import InsufficientCacheError, EpisodeDoneError
from ..utils import check_array
from ._nstep import NStep


class MockEnv:
    action_space = gym.spaces.Discrete(10)


class TestNStep:
    env = MockEnv()
    gamma = 0.85
    n = 5

    # rnd = jnp.random.RandomState(42)
    # S = jnp.arange(13)
    # A = rnd.randint(10, size=13)
    # R = rnd.randn(13)
    # D = jnp.zeros(13, dtype='bool')
    # D[-1] = True
    # In = (gamma ** n) * jnp.ones(13, dtype='bool')
    # In[-n:] = 0

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
    In = jnp.array([0.44370531249999995] * 8 + [0.0] * 5)
    episode = list(zip(S, A, R, D))

    @property
    def Rn(self):
        Rn_ = jnp.zeros_like(self.R)
        gammas = jnp.power(self.gamma, jnp.arange(13))
        for i in range(len(Rn_)):
            Rn_ = Rn_.at[i].set(
                self.R[i:(i + self.n)].dot(gammas[:len(self.R[i:(i + self.n)])]))
        return Rn_

    def test_append_done_twice(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            if i == 1:
                cache.add(s, a, r, True)
            else:
                with pytest.raises(EpisodeDoneError):
                    cache.add(s, a, r, True)

    def test_append_done_one(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            if i == 1:
                cache.add(s, a, r, True)
            else:
                break

        assert cache
        transitions = cache.flush()
        assert_array_almost_equal(transitions.S, self.S[:1])
        assert_array_almost_equal(transitions.A, self.A[:1])
        assert_array_almost_equal(transitions.Rn, self.R[:1])
        assert_array_almost_equal(transitions.In, [0])

    def test_pop(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i
            if i <= self.n:
                assert not cache
            if i > self.n:
                assert cache

        i = 0
        while cache:
            transition = cache.pop()
            check_array(transition.S, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.A, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.Rn, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.In, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.S_next, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.A_next, ndim=1, axis_size=1, axis=0, except_np=True)
            assert_array_almost_equal(transition.S[0], self.S[i])
            assert_array_almost_equal(transition.A[0], self.A[i])
            assert_array_almost_equal(transition.Rn[0], self.Rn[i])
            assert_array_almost_equal(transition.In[0], self.In[i])
            if i < 13 - self.n:
                assert_array_almost_equal(
                    transition.S_next[0], self.S[i + self.n])
                assert_array_almost_equal(
                    transition.A_next[0], self.A[i + self.n])
            i += 1

    def test_pop_eager(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode):
            cache.add(s, a, r, done)
            assert len(cache) == min(i + 1, self.n + 1)

            if cache:
                assert i + 1 > self.n
                transition = cache.pop()
                check_array(transition.S, ndim=1, axis_size=1, axis=0, except_np=True)
                check_array(transition.A, ndim=1, axis_size=1, axis=0, except_np=True)
                check_array(transition.Rn, ndim=1, axis_size=1, axis=0, except_np=True)
                check_array(transition.In, ndim=1, axis_size=1, axis=0, except_np=True)
                check_array(transition.S_next, ndim=1, axis_size=1, axis=0, except_np=True)
                check_array(transition.A_next, ndim=1, axis_size=1, axis=0, except_np=True)
                assert_array_almost_equal(transition.S[0], self.S[i - self.n])
                assert_array_almost_equal(transition.A[0], self.A[i - self.n])
                assert_array_almost_equal(
                    transition.Rn[0], self.Rn[i - self.n])
                assert_array_almost_equal(
                    transition.In[0], self.In[i - self.n])
                assert_array_almost_equal(transition.S_next[0], self.S[i])
                assert_array_almost_equal(transition.A_next[0], self.A[i])
            else:
                assert i + 1 <= self.n

        i = 13 - self.n
        while cache:
            transition = cache.pop()
            check_array(transition.S, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.A, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.Rn, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.In, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.S_next, ndim=1, axis_size=1, axis=0, except_np=True)
            check_array(transition.A_next, ndim=1, axis_size=1, axis=0, except_np=True)
            assert_array_almost_equal(transition.S[0], self.S[i])
            assert_array_almost_equal(transition.A[0], self.A[i])
            assert_array_almost_equal(transition.Rn[0], self.Rn[i])
            assert_array_almost_equal(transition.In[0], self.In[i])
            if i < 13 - self.n:
                assert_array_almost_equal(
                    transition.S_next[0], self.S[i + self.n])
                assert_array_almost_equal(
                    transition.A_next[0], self.A[i + self.n])
            i += 1

    def test_flush(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i
            if i <= self.n:
                assert not cache
            if i > self.n:
                assert cache

        transitions = cache.flush()
        assert_array_almost_equal(transitions.S, self.S)
        assert_array_almost_equal(transitions.A, self.A)
        assert_array_almost_equal(transitions.Rn, self.Rn)
        assert_array_almost_equal(transitions.In, self.In)
        assert_array_almost_equal(
            transitions.S_next[:-self.n], self.S[self.n:])
        assert_array_almost_equal(
            transitions.A_next[:-self.n], self.A[self.n:])

    def test_flush_eager(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in enumerate(self.episode):
            cache.add(s, a, r, done)
            assert len(cache) == min(i + 1, self.n + 1)

            if cache:
                assert i + 1 > self.n
                transitions = cache.flush()
                if i == 12:
                    slc = slice(i - self.n, None)
                    assert_array_almost_equal(transitions.S, self.S[slc])
                    assert_array_almost_equal(transitions.A, self.A[slc])
                    assert_array_almost_equal(transitions.Rn, self.Rn[slc])
                    assert_array_almost_equal(transitions.In, self.In[slc])
                    assert_array_almost_equal(
                        transitions.S_next.shape, (self.n + 1,))
                    assert_array_almost_equal(
                        transitions.A_next.shape, (self.n + 1,))
                else:
                    slc = slice(i - self.n, i - self.n + 1)
                    slc_next = slice(i, i + 1)
                    assert_array_almost_equal(transitions.S, self.S[slc])
                    assert_array_almost_equal(transitions.A, self.A[slc])
                    assert_array_almost_equal(transitions.Rn, self.Rn[slc])
                    assert_array_almost_equal(transitions.In, self.In[slc])
                    assert_array_almost_equal(transitions.S_next, self.S[slc_next])
                    assert_array_almost_equal(transitions.A_next, self.A[slc_next])
            else:
                assert i + 1 <= self.n

        i = 13 - self.n
        while cache:
            transition = cache.flush()
            assert transition.S == self.S[i]
            assert_array_almost_equal(a, self.A[i])
            assert transition.Rn == self.Rn[i]
            assert transition.In == self.In[i]
            if i < 13 - self.n:
                assert transition.S_next == self.S[i + self.n]
                assert transition.A_next == self.A[i + self.n]
            i += 1

    def test_flush_insufficient(self):
        cache = NStep(self.n, gamma=self.gamma)
        for i, (s, a, r, done) in islice(enumerate(self.episode, 1), 4):
            cache.add(s, a, r, done)

        with pytest.raises(InsufficientCacheError):
            cache.flush()

    def test_flush_empty(self):
        cache = NStep(self.n, gamma=self.gamma)

        with pytest.raises(InsufficientCacheError):
            cache.flush()

    def test_extra_info(self):
        cache = NStep(self.n, gamma=self.gamma, record_extra_info=True)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            cache.add(s, a, r, done)
            assert len(cache) == i
            if i <= self.n:
                assert not cache
            if i > self.n:
                assert cache

        transitions = cache.flush()
        assert type(transitions.extra_info) == dict
        states = jnp.stack(transitions.extra_info['states'])
        actions = jnp.stack(transitions.extra_info['actions'])
        assert_array_almost_equal(states[0], transitions.S)
        assert_array_almost_equal(actions[0], transitions.A)

    def test_extra_info_dones(self):
        cache = NStep(self.n, gamma=self.gamma, record_extra_info=True)
        for i, (s, a, r, done) in enumerate(self.episode, 1):
            if i == self.n + 2:
                cache.add(s, a, r, True)
                break
            else:
                cache.add(s, a, r, False)

        assert cache
        transitions = cache.flush()
        assert type(transitions.extra_info) == dict
        dones = jnp.stack(transitions.extra_info['dones'])
        for i in range(self.n + 2):
            assert dones[:, i].sum() == i
