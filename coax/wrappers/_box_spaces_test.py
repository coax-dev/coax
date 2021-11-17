from itertools import combinations

import gym
import numpy as onp

from .._base.test_case import TestCase
from ._box_spaces import BoxActionsToDiscrete


class TestBoxActionsToDiscrete(TestCase):
    def test_inverse(self):
        num_bins = 100

        env = gym.make('BipedalWalker-v3')
        env = BoxActionsToDiscrete(env, num_bins)

        hi, lo = env.env.action_space.high, env.env.action_space.low
        a_orig = env.env.action_space.sample()

        # create discrete action
        a_orig_rescaled = (a_orig - lo) / (hi - lo)
        a_orig_flat = onp.ravel(a_orig_rescaled)
        a_discrete = onp.asarray(num_bins * a_orig_flat, dtype='int8')

        # reconstruct continuous action
        a_reconstructed = env._discrete_to_box(a_discrete)

        diff = onp.abs(a_reconstructed - a_orig) / (hi - lo)
        print(diff)
        self.assertTrue(onp.all(diff < 1 / num_bins))

    def test_not_all_same(self):
        env = gym.make('BipedalWalker-v3')
        env = BoxActionsToDiscrete(env, num_bins=10, random_seed=13)

        a_discrete = onp.zeros(4)
        a_box = env._discrete_to_box(a_discrete)
        print(a_box)

        for x, y in combinations(a_box, 2):
            self.assertNotAlmostEqual(x, y)
