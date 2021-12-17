import jax.numpy as jnp

from .._base.test_case import TestCase
from ._log_transform import LogTransform


class TestLogTransform(TestCase):
    decimal = 5

    def test_inverse(self):
        f = LogTransform(scale=7)
        # some consistency checks
        values = jnp.array([-100, -10, -1, 0, 1, 10, 100], dtype='float32')
        self.assertArrayAlmostEqual(
            f.inverse_func(f.transform_func(values)), values)
