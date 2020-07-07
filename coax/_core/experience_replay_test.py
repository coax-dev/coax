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
from gym.wrappers.frame_stack import LazyFrames
from numpy.testing import assert_array_almost_equal
from jax import numpy as jnp
from jax.ops import index_update

from .experience_replay import ExperienceReplayBuffer


class MockEnv:
    action_space = gym.spaces.Discrete(7)


class TestExperienceReplayBuffer:
    N = 7
    S = jnp.expand_dims(jnp.arange(N), axis=1)
    A = S[:, 0] % 100
    R = S[:, 0]
    D = jnp.zeros(N, dtype='bool')
    D = index_update(D, -1, True)
    EPISODE = list(zip(S, A, R, D))

    def test_add(self):
        # in this test method we don't do frame stacking
        buffer = ExperienceReplayBuffer(MockEnv, capacity=17)
        for i, (s, a, r, done) in enumerate(self.EPISODE, 1):
            buffer.add(s + 100, a, r + 100, done)
            assert len(buffer) == max(0, i - buffer.n)

        assert_array_almost_equal(
            buffer._e[:7],
            [0, 0, 0, 0, 0, 0, 0])
        assert_array_almost_equal(
            buffer._d[:7].astype('int32'),
            [0, 0, 0, 0, 0, 0, 1])

        for i, (s, a, r, done) in enumerate(self.EPISODE, i + 1):
            buffer.add(s + 200, a, r + 200, done)
            assert len(buffer) == max(0, i - buffer.n)

        assert_array_almost_equal(
            buffer._e[:14],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        assert_array_almost_equal(
            buffer._d[:14].astype('int32'),
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

        for i, (s, a, r, done) in enumerate(self.EPISODE, i + 1):
            buffer.add(s + 300, a, r + 300, done)
            assert len(buffer) == jnp.clip(i - buffer.n, 0, 17)

        # buffer wraps around and overwrites oldest transitions
        assert_array_almost_equal(
            buffer._e,
            [2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        assert_array_almost_equal(
            buffer._d.astype('int32'),
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        assert_array_almost_equal(
            buffer._a,
            [4, 5, 6, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3])

    def test_sample(self):
        num_stack = 4
        buffer = ExperienceReplayBuffer(
            env=MockEnv, capacity=17, random_seed=7, n=2)

        for ep in (1, 2, 3):
            for s, a, r, done in self.EPISODE:
                s = LazyFrames([s[[0, 0, 0, 0]] + ep * 100] * num_stack)
                buffer.add(s, a, r + ep * 100, done)

        # quickly check content, just to be safe
        assert_array_almost_equal(
            buffer._a,
            [5, 6, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4])

        expected_transitions = [
            [[300, 301, 302, 303], 0.9801, [302, 303, 304, 305]],  # normal
            [[300, 300, 300, 300], 0.9801, [300, 300, 301, 302]],  # fill both
            [[300, 300, 300, 300], 0.9801, [300, 300, 301, 302]],  # fill both
            [[300, 301, 302, 303], 0.9801, [302, 303, 304, 305]],  # normal
            [[300, 300, 301, 302], 0.9801, [301, 302, 303, 304]],  # fill left
            [[302, 303, 304, 305], 0.0000, [304, 305, 306, 102]],  # end of ep
            [[103, 104, 105, 106], 0.0000, [105, 106, 200, 201]],  # end of ep
            [[302, 303, 304, 305], 0.0000, [304, 305, 306, 102]],  # end of ep
            [[201, 202, 203, 204], 0.9801, [203, 204, 205, 206]],  # normal
            [[203, 204, 205, 206], 0.0000, [205, 206, 300, 301]],  # end of ep
            [[200, 200, 201, 202], 0.9801, [201, 202, 203, 204]],  # fill left
            [[300, 300, 301, 302], 0.9801, [301, 302, 303, 304]],  # fill left
            [[201, 202, 203, 204], 0.9801, [203, 204, 205, 206]],  # normal
            [[300, 300, 300, 301], 0.9801, [300, 301, 302, 303]],  # fill left
            [[300, 301, 302, 303], 0.9801, [302, 303, 304, 305]],  # normal
            [[300, 301, 302, 303], 0.9801, [302, 303, 304, 305]],  # normal
        ]

        transitions = buffer.sample(batch_size=16)
        actual_transitions = [list(tup) for tup in zip(
            transitions.S[:, :, 0],
            transitions.In,
            transitions.S_next[:, :, 0])]
        comments = set()
        for a, b, c in actual_transitions:
            if c[0] == c[1]:
                comment = 'fill both'
            elif a[0] == a[1]:
                comment = 'fill left'
            elif (c[-1] // 100) != (a[-1] // 100):
                comment = 'end of ep'
            else:
                comment = 'normal'
            comments.add(comment)
            print(f"            [{list(a)}, {b:.4f}, {list(c)}],  # {comment}")
        assert len(comments) == 4
        assert_array_almost_equal(
            [tr[1] for tr in actual_transitions],
            [tr[1] for tr in expected_transitions],
            decimal=5)
        assert_array_almost_equal(
            [tr[0] for tr in actual_transitions],
            [tr[0] for tr in expected_transitions])
        assert_array_almost_equal(
            [tr[2] for tr in actual_transitions],
            [tr[2] for tr in expected_transitions])

        # check if actions are separate by n steps
        for a, i_next, a_next in zip(transitions.A, transitions.In, transitions.A_next):  # noqa: E501
            if i_next != 0:
                assert a_next - a == buffer.n

        # check if states and actions are aligned
        assert_array_almost_equal(
            transitions.S[:, -1, 0] % 100,
            transitions.A)
        assert_array_almost_equal(
            transitions.S_next[:, -1, 0] % 100,
            transitions.A_next)

    def test_shape(self):
        num_stack = 3
        buffer = ExperienceReplayBuffer(
            env=MockEnv, capacity=17, random_seed=5)

        for ep in (1, 2, 3):
            for i, (_, a, r, done) in enumerate(self.EPISODE):
                s = 100 * ep + i * jnp.ones((11, 13), dtype='int32')
                s = LazyFrames([s] * num_stack)
                buffer.add(s, a, r, done)

        transitions = buffer.sample(batch_size=5)
        assert transitions.S.shape == (5, 3, 11, 13)

        for row in transitions.S[:, :, 0, 0]:
            print(f"                {list(row)},")

        # check if all frames come from the same episode
        assert_array_almost_equal(
            # look at upper-left pixel only
            transitions.S[:, :, 0, 0], [
                [300, 300, 300],
                [200, 200, 201],
                [304, 305, 306],
                [103, 104, 105],
                [202, 203, 204],
            ])
