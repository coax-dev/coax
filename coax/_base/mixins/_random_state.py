import numpy as onp
import jax


class RandomStateMixin:
    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, new_random_seed):
        if new_random_seed is None:
            new_random_seed = onp.random.randint(2147483647)
        self._random_seed = new_random_seed
        self._random_key = jax.random.PRNGKey(self._random_seed)

    @property
    def rng(self):
        self._random_key, key = jax.random.split(self._random_key)
        return key
