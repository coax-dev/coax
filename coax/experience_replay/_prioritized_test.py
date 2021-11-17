from copy import deepcopy

import gym
import pytest
import numpy as onp

from ..utils import get_transition_batch
from ._simple import SimpleReplayBuffer
from ._prioritized import PrioritizedReplayBuffer


@pytest.mark.parametrize('n', [2, 4])  # 2 * batch_size < capacity, 4 * batch_size > capacity
def test_consistency(n):
    env = gym.make('FrozenLakeNonSlippery-v0')
    buffer1 = SimpleReplayBuffer(capacity=100)
    buffer2 = PrioritizedReplayBuffer(capacity=100)

    for i in range(n):
        transition_batch = get_transition_batch(env, batch_size=32, random_seed=i)
        buffer1.add(transition_batch)
        buffer2.add(transition_batch, Adv=transition_batch.Rn)

    # test TransitionBatch.__eq__
    assert buffer1._storage[0] == buffer1._storage[0]
    assert buffer1._storage[0] != buffer1._storage[1]

    # test consistency between two buffers
    assert len(buffer1) == len(buffer2)
    for i in range(len(buffer1)):
        t1 = buffer1._storage[i]
        t2 = buffer2._storage[(i + buffer2._index) % len(buffer2)]
        assert t1 == t2


def test_sample():
    env = gym.make('FrozenLakeNonSlippery-v0')
    buffer1 = SimpleReplayBuffer(capacity=100, random_seed=13)
    buffer2 = PrioritizedReplayBuffer(capacity=100, random_seed=13)

    for i in range(4):
        transition_batch = get_transition_batch(env, batch_size=32, random_seed=i)
        transition_batch.W = onp.ones_like(transition_batch.W)  # start with uniform weights
        buffer1.add(transition_batch)
        buffer2.add(transition_batch.copy(), Adv=transition_batch.Rn)

    assert onp.allclose(buffer1.sample(batch_size=10).W, onp.ones(10))
    assert not onp.allclose(buffer2.sample(batch_size=10).W, onp.ones(10))


def test_alpha():
    env = gym.make('FrozenLakeNonSlippery-v0')
    buffer = PrioritizedReplayBuffer(capacity=100, random_seed=13, alpha=0.8)

    for i in range(4):
        transition_batch = get_transition_batch(env, batch_size=32, random_seed=i)
        transition_batch.W = onp.ones_like(transition_batch.W)  # start with uniform weights
        buffer.add(transition_batch.copy(), Adv=transition_batch.Rn)

    with pytest.raises(TypeError):
        buffer.alpha = 'foo'

    with pytest.raises(TypeError):
        buffer.alpha = 0.

    with pytest.raises(TypeError):
        buffer.alpha = -1.

    assert onp.isclose(buffer.alpha, 0.8)
    old_values = deepcopy(buffer._sumtree.values)
    old_alpha = buffer.alpha

    buffer.alpha *= 1.5
    assert onp.isclose(buffer.alpha, 1.2)
    new_values = buffer._sumtree.values
    new_alpha = buffer.alpha

    diff = 2 * (new_values - old_values) / (new_values + old_values)
    assert onp.min(onp.abs(diff)) > 1e-3
    assert onp.allclose(onp.power(new_values, 1 / new_alpha), onp.power(old_values, 1 / old_alpha))


def test_update():
    env = gym.make('FrozenLakeNonSlippery-v0')
    buffer = PrioritizedReplayBuffer(capacity=100, random_seed=13, alpha=0.8)

    for i in range(buffer.capacity):
        transition_batch = get_transition_batch(env, random_seed=i)
        transition_batch.W = onp.ones_like(transition_batch.W)  # start with uniform weights
        buffer.add(transition_batch.copy(), Adv=transition_batch.Rn)

    t = buffer.sample(50)
    old_values = deepcopy(buffer._sumtree.values)
    print(t)

    # add more transitions after sampling
    for i in range(buffer.capacity // 2):
        transition_batch = get_transition_batch(env, random_seed=(buffer.capacity + i))
        transition_batch.W = onp.ones_like(transition_batch.W)  # start with uniform weights
        buffer.add(transition_batch.copy(), Adv=transition_batch.Rn)

    # update values where some transitions in buffer have been replaced by newer transitions
    Adv_new = get_transition_batch(env, batch_size=t.batch_size, random_seed=7).Rn
    buffer.update(t.idx, Adv_new)
    new_values = buffer._sumtree.values

    # all values in first half are replaced by buffer.add()
    r = slice(None, 50)
    diff = 2 * (new_values[r] - old_values[r]) / (new_values[r] + old_values[r])
    print(onp.abs(diff))
    assert onp.min(onp.abs(diff)) > 1e-3

    # not replaced by buffer.add(), but updated by buffer.update()
    u = t.idx[t.idx >= 50]
    diff = 2 * (new_values[u] - old_values[u]) / (new_values[u] + old_values[u])
    print(onp.abs(diff))
    assert onp.min(onp.abs(diff)) > 1e-4

    # neither replaced by buffer.add() nor updated by buffer.update()
    n = ~onp.isin(onp.arange(100), t.idx) & (onp.arange(100) >= 50)
    print(new_values[n])
    print(old_values[n])
    assert onp.sum(n) > 0
    assert onp.allclose(new_values[n], old_values[n])
