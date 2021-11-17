from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from gym.spaces import Space

from ..utils import safe_sample, batch_to_single, jit
from .base_func import BaseFunc, ExampleData, Inputs, ArgsType2


__all__ = (
    'BaseStochasticFuncType2',
    'StochasticFuncType2Mixin',
)


class StochasticFuncType2Mixin:
    r"""

    An mix-in class for stochastic functions that take *only states* as input:

    - Policy
    - StochasticV
    - StateDensity

    """
    def __call__(self, s, return_logp=False):
        S = self.observation_preprocessor(self.rng, s)
        X, logP = self.sample_func(self.params, self.function_state, self.rng, S)
        x = self.proba_dist.postprocess_variate(self.rng, X)
        return (x, batch_to_single(logP)) if return_logp else x

    def mean(self, s):
        S = self.observation_preprocessor(self.rng, s)
        X = self.mean_func(self.params, self.function_state, self.rng, S)
        x = self.proba_dist.postprocess_variate(self.rng, X)
        return x

    def mode(self, s):
        S = self.observation_preprocessor(self.rng, s)
        X = self.mode_func(self.params, self.function_state, self.rng, S)
        x = self.proba_dist.postprocess_variate(self.rng, X)
        return x

    def dist_params(self, s):
        S = self.observation_preprocessor(self.rng, s)
        dist_params, _ = self.function(self.params, self.function_state, self.rng, S, False)
        return batch_to_single(dist_params)

    @property
    def sample_func(self):
        r"""

        The function that is used for sampling *random* from the underlying :attr:`proba_dist`,
        defined as a JIT-compiled pure function. This function may be called directly as:

        .. code:: python

            output = obj.sample_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_sample_func'):
            def sample_func(params, state, rng, S):
                rngs = hk.PRNGSequence(rng)
                dist_params, _ = self.function(params, state, next(rngs), S, False)
                X = self.proba_dist.sample(dist_params, next(rngs))
                logP = self.proba_dist.log_proba(dist_params, X)
                return X, logP
            self._sample_func = jit(sample_func)
        return self._sample_func

    @property
    def mean_func(self):
        r"""

        The function that is used for getting the mean of the distribution, defined as a
        JIT-compiled pure function. This function may be called directly as:

        .. code:: python

            output = obj.mean_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_mean_func'):
            def mean_func(params, state, rng, S):
                dist_params, _ = self.function(params, state, rng, S, False)
                return self.proba_dist.mean(dist_params)
            self._mean_func = jit(mean_func)
        return self._mean_func

    @property
    def mode_func(self):
        r"""

        The function that is used for getting the mode of the distribution, defined as a
        JIT-compiled pure function. This function may be called directly as:

        .. code:: python

            output = obj.mode_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_mode_func'):
            def mode_func(params, state, rng, S):
                dist_params, _ = self.function(params, state, rng, S, False)
                return self.proba_dist.mode(dist_params)
            self._mode_func = jit(mode_func)
        return self._mode_func


class BaseStochasticFuncType2(BaseFunc, StochasticFuncType2Mixin):
    r"""

    An abstract base class for stochastic function that take *only states* as input:

    - Policy
    - StochasticV
    - StateDensity

    """
    def __init__(
            self, func, observation_space, action_space, observation_preprocessor, proba_dist,
            random_seed):

        self.observation_preprocessor = observation_preprocessor
        self.proba_dist = proba_dist

        # note: self._modeltype is set in super().__init__ via self._check_signature
        super().__init__(
            func=func,
            observation_space=observation_space,
            action_space=action_space,
            random_seed=random_seed)

    @classmethod
    def example_data(
            cls, env, observation_preprocessor=None, proba_dist=None,
            batch_size=1, random_seed=None):

        if not isinstance(env.observation_space, Space):
            raise TypeError(
                "env.observation_space must be derived from gym.Space, "
                f"got: {type(env.observation_space)}")
        if not isinstance(env.action_space, Space):
            raise TypeError(
                f"env.action_space must be derived from gym.Space, got: {type(env.action_space)}")

        # these must be provided
        assert observation_preprocessor is not None
        assert proba_dist is not None

        rnd = onp.random.RandomState(random_seed)
        rngs = hk.PRNGSequence(rnd.randint(jnp.iinfo('int32').max))

        # input: state observations
        S = [safe_sample(env.observation_space, rnd) for _ in range(batch_size)]
        S = [observation_preprocessor(next(rngs), s) for s in S]
        S = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *S)

        # output
        dist_params = jax.tree_map(
            lambda x: jnp.asarray(rnd.randn(batch_size, *x.shape[1:])), proba_dist.default_priors)

        return ExampleData(
            inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
            output=dist_params,
        )

    def _check_signature(self, func):
        if tuple(signature(func).parameters) != ('S', 'is_training'):
            sig = ', '.join(signature(func).parameters)
            raise TypeError(
                f"func has bad signature; expected: func(S, is_training), got: func({sig})")

        Env = namedtuple('Env', ('observation_space', 'action_space'))
        return BaseStochasticFuncType2.example_data(
            env=Env(self.observation_space, self.action_space),
            observation_preprocessor=self.observation_preprocessor,
            proba_dist=self.proba_dist,
            batch_size=1,
            random_seed=self.random_seed,
        )

    def _check_output(self, actual, expected):
        expected_leaves, expected_structure = jax.tree_flatten(expected)
        actual_leaves, actual_structure = jax.tree_flatten(actual)
        assert all(isinstance(x, jnp.ndarray) for x in expected_leaves), "bad example_data"

        if actual_structure != expected_structure:
            raise TypeError(
                f"func has bad return tree_structure, expected: {expected_structure}, "
                f"got: {actual_structure}")

        if not all(isinstance(x, jnp.ndarray) for x in actual_leaves):
            bad_types = tuple(type(x) for x in actual_leaves if not isinstance(x, jnp.ndarray))
            raise TypeError(
                "all leaves of dist_params must be of type: jax.numpy.ndarray, "
                f"found leaves of type: {bad_types}")

        if not all(a.shape == b.shape for a, b in zip(actual_leaves, expected_leaves)):
            shapes_tree = jax.tree_multimap(
                lambda a, b: f"{a.shape} {'!=' if a.shape != b.shape else '=='} {b.shape}",
                actual, expected)
            raise TypeError(f"found leaves with unexpected shapes: {shapes_tree}")
