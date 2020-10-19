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
    assert buffer1._deque[0] == buffer1._deque[0]
    assert buffer1._deque[0] != buffer1._deque[1]

    # test consistency between two buffers
    assert len(buffer1) == len(buffer2)
    for i in range(len(buffer1)):
        t1 = buffer1._deque[i]
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
