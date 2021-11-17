import jax.numpy as jnp

from .._base.test_case import TestCase
from ._action_noise import OrnsteinUhlenbeckNoise


class TestOrnsteinUhlenbeckNoise(TestCase):
    def test_overall_mean_variance(self):
        noise = OrnsteinUhlenbeckNoise(random_seed=13)
        x = jnp.stack([noise(0.) for _ in range(1000)])
        mu, sigma = jnp.mean(x), jnp.std(x)
        self.assertLess(abs(mu), noise.theta)
        self.assertGreater(sigma, noise.sigma)
        self.assertLess(sigma, noise.sigma * 2)
