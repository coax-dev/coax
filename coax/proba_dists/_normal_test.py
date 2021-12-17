import gym
import jax
import haiku as hk

from .._base.test_case import TestCase
from ._normal import NormalDist


class TestNormalDist(TestCase):
    decimal = 5

    def setUp(self):
        self.rngs = hk.PRNGSequence(13)

    def tearDown(self):
        del self.rngs

    def test_kl_divergence(self):
        dist = NormalDist(gym.spaces.Box(low=0, high=1, shape=(7,)))
        params_p = {
            'mu': jax.random.normal(next(self.rngs), shape=(3, 7)),
            'logvar': jax.random.normal(next(self.rngs), shape=(3, 7))}
        params_q = {
            'mu': jax.random.normal(next(self.rngs), shape=(3, 7)),
            'logvar': jax.random.normal(next(self.rngs), shape=(3, 7))}
        # params_q = {k: v + 0.001 for k, v in params_p.items()}

        kl_div_direct = dist.kl_divergence(params_p, params_q)
        kl_div_from_ce = dist.cross_entropy(params_p, params_q) - dist.entropy(params_p)
        self.assertArrayAlmostEqual(kl_div_direct, kl_div_from_ce)

    def test_box_clip(self):
        msg = (
            r"one or more dimensions of Box\(low=.*, high=.*\) "
            r"will be clipped to Box\(low=.*, high=.*\)"
        )
        with self.assertWarnsRegex(UserWarning, msg):
            dist = NormalDist(gym.spaces.Box(low=-1000, high=10000000, shape=(1,)))

        self.assertGreater(dist._low[0, 0], dist.space.low[0])
        self.assertLess(dist._high[0, 0], dist.space.high[0])
