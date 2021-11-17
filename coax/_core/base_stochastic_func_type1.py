from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from gym.spaces import Space, Discrete

from ..utils import safe_sample, batch_to_single, jit
from .base_func import BaseFunc, ExampleData, Inputs, ArgsType1, ArgsType2, ModelTypes


__all__ = (
    'BaseStochasticFuncType1',
)


class BaseStochasticFuncType1(BaseFunc):
    r"""

    An abstract base class for stochastic functions that take *state-action pairs* as input:

    - StochasticQ
    - StochasticTransitionModel
    - StochasticRewardFunction

    """
    def __init__(
            self, func, observation_space, action_space,
            observation_preprocessor, action_preprocessor, proba_dist, random_seed):

        self.observation_preprocessor = observation_preprocessor
        self.action_preprocessor = action_preprocessor
        self.proba_dist = proba_dist

        # note: self._modeltype is set in super().__init__ via self._check_signature
        super().__init__(
            func=func,
            observation_space=observation_space,
            action_space=action_space,
            random_seed=random_seed)

    def __call__(self, s, a=None, return_logp=False):
        S = self.observation_preprocessor(self.rng, s)
        if a is None:
            X, logP = self.sample_func_type2(self.params, self.function_state, self.rng, S)
            X, logP = batch_to_single((X, logP))  # (batch, num_actions, *) -> (num_actions, *)
            n = self.action_space.n
            x = [self.proba_dist.postprocess_variate(self.rng, X, index=i) for i in range(n)]
            logp = list(logP)
        else:
            A = self.action_preprocessor(self.rng, a)
            X, logP = self.sample_func_type1(self.params, self.function_state, self.rng, S, A)
            x = self.proba_dist.postprocess_variate(self.rng, X)
            logp = batch_to_single(logP)
        return (x, logp) if return_logp else x

    def mean(self, s, a=None):
        S = self.observation_preprocessor(self.rng, s)
        if a is None:
            X = self.mean_func_type2(self.params, self.function_state, self.rng, S)
            X = batch_to_single(X)  # (batch, num_actions, *) -> (num_actions, *)
            n = self.action_space.n
            x = [self.proba_dist.postprocess_variate(self.rng, X, index=i) for i in range(n)]
        else:
            A = self.action_preprocessor(self.rng, a)
            X = self.mean_func_type1(self.params, self.function_state, self.rng, S, A)
            x = self.proba_dist.postprocess_variate(self.rng, X)
        return x

    def mode(self, s, a=None):
        S = self.observation_preprocessor(self.rng, s)
        if a is None:
            X = self.mode_func_type2(self.params, self.function_state, self.rng, S)
            X = batch_to_single(X)  # (batch, num_actions, *) -> (num_actions, *)
            n = self.action_space.n
            x = [self.proba_dist.postprocess_variate(self.rng, X, index=i) for i in range(n)]
        else:
            A = self.action_preprocessor(self.rng, a)
            X = self.mode_func_type1(self.params, self.function_state, self.rng, S, A)
            x = self.proba_dist.postprocess_variate(self.rng, X)
        return x

    def dist_params(self, s, a=None):
        S = self.observation_preprocessor(self.rng, s)
        if a is None:
            # batch_to_single() projects: (batch, num_actions, *shape) -> (num_actions, *shape)
            dist_params = batch_to_single(
                self.function_type2(self.params, self.function_state, self.rng, S, False)[0])
            dist_params = [
                batch_to_single(dist_params, index=i) for i in range(self.action_space.n)]
        else:
            A = self.action_preprocessor(self.rng, a)
            dist_params, _ = self.function_type1(
                self.params, self.function_state, self.rng, S, A, False)
            dist_params = batch_to_single(dist_params)
        return dist_params

    @property
    def function_type1(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-1 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 1:
            return self.function

        assert isinstance(self.action_space, Discrete)

        def project(A):
            assert A.ndim == 2, f"bad shape: {A.shape}"
            assert A.shape[1] == self.action_space.n, f"bad shape: {A.shape}"
            def func(leaf):  # noqa: E306
                assert isinstance(leaf, jnp.ndarray), f"leaf must be ndarray, got: {type(leaf)}"
                assert leaf.ndim >= 2, f"bad shape: {leaf.shape}"
                assert leaf.shape[0] == A.shape[0], \
                    f"batch_size (axis=0) mismatch: leaf.shape: {leaf.shape}, A.shape: {A.shape}"
                assert leaf.shape[1] == A.shape[1], \
                    f"num_actions (axis=1) mismatch: leaf.shape: {leaf.shape}, A.shape: {A.shape}"
                return jax.vmap(jnp.dot)(jnp.moveaxis(leaf, 1, -1), A)
            return func

        def type1_func(type2_params, type2_state, rng, S, A, is_training):
            dist_params, state_new = self.function(type2_params, type2_state, rng, S, is_training)
            dist_params = jax.tree_map(project(A), dist_params)
            return dist_params, state_new

        return type1_func

    @property
    def function_type2(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-2 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 2:
            return self.function

        if not isinstance(self.action_space, Discrete):
            raise ValueError(
                "input 'A' is required for type-1 dynamics model when action space is non-Discrete")

        n = self.action_space.n

        def reshape(leaf):
            # reshape from (batch * num_actions, *shape) -> (batch, *shape, num_actions)
            assert isinstance(leaf, jnp.ndarray), f"all leaves must be ndarray, got: {type(leaf)}"
            assert leaf.ndim >= 1, f"bad shape: {leaf.shape}"
            assert leaf.shape[0] % n == 0, \
                f"first axis size must be a multiple of num_actions, got shape: {leaf.shape}"
            leaf = jnp.reshape(leaf, (-1, n, *leaf.shape[1:]))  # (batch, num_actions, *shape)
            return leaf

        def type2_func(type1_params, type1_state, rng, S, is_training):
            rngs = hk.PRNGSequence(rng)
            batch_size = jax.tree_leaves(S)[0].shape[0]

            # example: let S = [7, 2, 5, 8] and num_actions = 3, then
            # S_rep = [7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8]  # repeated
            # A_rep = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # tiled
            S_rep = jax.tree_map(lambda x: jnp.repeat(x, n, axis=0), S)
            A_rep = jnp.tile(jnp.arange(n), batch_size)
            A_rep = self.action_preprocessor(next(rngs), A_rep)  # one-hot encoding

            # evaluate on replicas => output shape: (batch * num_actions, *shape)
            dist_params_rep, state_new = self.function(
                type1_params, type1_state, next(rngs), S_rep, A_rep, is_training)
            dist_params = jax.tree_map(reshape, dist_params_rep)

            return dist_params, state_new

        return type2_func

    @property
    def modeltype(self):
        r"""

        Specifier for how the dynamics model is implemented, i.e.

        .. math::

            (s,a)   &\mapsto p(s'|s,a)  &\qquad (\text{modeltype} &= 1) \\
            s       &\mapsto p(s'|s,.)  &\qquad (\text{modeltype} &= 2)

        Note that modeltype=2 is only well-defined if the action space is :class:`Discrete
        <gym.spaces.Discrete>`. Namely, :math:`n` is the number of discrete actions.

        """
        return self._modeltype

    @classmethod
    def example_data(
            cls, env, observation_preprocessor, action_preprocessor, proba_dist,
            batch_size=1, random_seed=None):

        if not isinstance(env.observation_space, Space):
            raise TypeError(
                "env.observation_space must be derived from gym.Space, "
                f"got: {type(env.observation_space)}")
        if not isinstance(env.action_space, Space):
            raise TypeError(
                f"env.action_space must be derived from gym.Space, got: {type(env.action_space)}")

        rnd = onp.random.RandomState(random_seed)
        rngs = hk.PRNGSequence(rnd.randint(jnp.iinfo('int32').max))

        # these must be provided
        assert observation_preprocessor is not None
        assert action_preprocessor is not None
        assert proba_dist is not None

        # input: state observations
        S = [safe_sample(env.observation_space, rnd) for _ in range(batch_size)]
        S = [observation_preprocessor(next(rngs), s) for s in S]
        S = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *S)

        # input: actions
        A = [safe_sample(env.action_space, rnd) for _ in range(batch_size)]
        A = [action_preprocessor(next(rngs), a) for a in A]
        A = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *A)

        # output: type1
        dist_params_type1 = jax.tree_map(
            lambda x: jnp.asarray(rnd.randn(batch_size, *x.shape[1:])), proba_dist.default_priors)
        data_type1 = ExampleData(
            inputs=Inputs(args=ArgsType1(S=S, A=A, is_training=True), static_argnums=(2,)),
            output=dist_params_type1)

        if not isinstance(env.action_space, Discrete):
            return ModelTypes(type1=data_type1, type2=None)

        # output: type2 (if actions are discrete)
        dist_params_type2 = jax.tree_map(
            lambda x: jnp.asarray(rnd.randn(batch_size, env.action_space.n, *x.shape[1:])),
            proba_dist.default_priors)
        data_type2 = ExampleData(
            inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
            output=dist_params_type2)

        return ModelTypes(type1=data_type1, type2=data_type2)

    @property
    def sample_func_type1(self):
        r"""

        The function that is used for generating *random samples*, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.sample_func_type1(obj.params, obj.function_state, obj.rng, S)

        """
        if not hasattr(self, '_sample_func_type1'):
            def sample_func_type1(params, state, rng, S, A):
                rngs = hk.PRNGSequence(rng)
                dist_params, _ = self.function_type1(params, state, next(rngs), S, A, False)
                S_next = self.proba_dist.sample(dist_params, next(rngs))
                logP = self.proba_dist.log_proba(dist_params, S_next)
                return S_next, logP
            self._sample_func_type1 = jit(sample_func_type1)
        return self._sample_func_type1

    @property
    def sample_func_type2(self):
        r"""

        The function that is used for generating *random samples*, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.sample_func_type2(obj.params, obj.function_state, obj.rng, S, A)

        """
        if not hasattr(self, '_sample_func_type2'):
            def sample_func_type2(params, state, rng, S):
                rngs = hk.PRNGSequence(rng)
                dist_params, _ = self.function_type2(params, state, next(rngs), S, False)
                dist_params = jax.tree_map(self._reshape_to_replicas, dist_params)
                X = self.proba_dist.sample(dist_params, next(rngs))    # (batch x n, *shape)
                logP = self.proba_dist.log_proba(dist_params, X)       # (batch x n)
                X = jax.tree_map(self._reshape_from_replicas, X)       # (batch, n, *shape)
                logP = self._reshape_from_replicas(logP)               # (batch, n)
                return X, logP
            self._sample_func_type2 = jit(sample_func_type2)
        return self._sample_func_type2

    @property
    def mode_func_type1(self):
        r"""

        The function that is used for computing the *mode*, defined as a JIT-compiled pure function.
        This function may be called directly as:

        .. code:: python

            output = obj.mode_func_type1(obj.params, obj.function_state, obj.rng, S, A)

        """
        if not hasattr(self, '_mode_func_type1'):
            def mode_func_type1(params, state, rng, S, A):
                dist_params, _ = self.function_type1(params, state, rng, S, A, False)
                X = self.proba_dist.mode(dist_params)
                return X
            self._mode_func_type1 = jit(mode_func_type1)
        return self._mode_func_type1

    @property
    def mode_func_type2(self):
        r"""

        The function that is used for computing the *mode*, defined as a JIT-compiled pure function.
        This function may be called directly as:

        .. code:: python

            output = obj.mode_func_type2(obj.params, obj.function_state, obj.rng, S)

        """
        if not hasattr(self, '_mode_func_type2'):
            def mode_func_type2(params, state, rng, S):
                dist_params, _ = self.function_type2(params, state, rng, S, False)
                dist_params = jax.tree_map(self._reshape_to_replicas, dist_params)
                X = self.proba_dist.mode(dist_params)             # (batch x n, *shape)
                X = jax.tree_map(self._reshape_from_replicas, X)  # (batch, n, *shape)
                return X
            self._mode_func_type2 = jit(mode_func_type2)
        return self._mode_func_type2

    @property
    def mean_func_type1(self):
        r"""

        The function that is used for computing the *mean*, defined as a JIT-compiled pure function.
        This function may be called directly as:

        .. code:: python

            output = obj.mean_func_type1(obj.params, obj.function_state, obj.rng, S, A)

        """
        if not hasattr(self, '_mean_func_type1'):
            def mean_func_type1(params, state, rng, S, A):
                dist_params, _ = self.function_type1(params, state, rng, S, A, False)
                X = self.proba_dist.mean(dist_params)
                return X
            self._mean_func_type1 = jit(mean_func_type1)
        return self._mean_func_type1

    @property
    def mean_func_type2(self):
        r"""

        The function that is used for computing the *mean*, defined as a JIT-compiled pure function.
        This function may be called directly as:

        .. code:: python

            output = obj.mean_func_type2(obj.params, obj.function_state, obj.rng, S)

        """
        if not hasattr(self, '_mean_func_type2'):
            def mean_func_type2(params, state, rng, S):
                dist_params, _ = self.function_type2(params, state, rng, S, False)
                dist_params = jax.tree_map(self._reshape_to_replicas, dist_params)
                X = self.proba_dist.mean(dist_params)             # (batch x n, *shape)
                X = jax.tree_map(self._reshape_from_replicas, X)  # (batch, n, *shape)
                return X
            self._mean_func_type2 = jit(mean_func_type2)
        return self._mean_func_type2

    def _check_signature(self, func):
        sig_type1 = ('S', 'A', 'is_training')
        sig_type2 = ('S', 'is_training')
        sig = tuple(signature(func).parameters)

        if sig not in (sig_type1, sig_type2):
            sig = ', '.join(sig)
            alt = ' or func(S, is_training)' if isinstance(self.action_space, Discrete) else ''
            raise TypeError(
                f"func has bad signature; expected: func(S, A, is_training){alt}, got: func({sig})")

        if sig == sig_type2 and not isinstance(self.action_space, Discrete):
            raise TypeError(
                "type-2 models are only well-defined for Discrete action spaces")

        Env = namedtuple('Env', ('observation_space', 'action_space'))
        example_data_per_modeltype = BaseStochasticFuncType1.example_data(
            env=Env(self.observation_space, self.action_space),
            observation_preprocessor=self.observation_preprocessor,
            action_preprocessor=self.action_preprocessor,
            proba_dist=self.proba_dist,
            batch_size=1,
            random_seed=self.random_seed)

        if sig == sig_type1:
            self._modeltype = 1
            example_data = example_data_per_modeltype.type1
        else:
            self._modeltype = 2
            example_data = example_data_per_modeltype.type2

        return example_data

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

    def _reshape_from_replicas(self, leaf):
        """ reshape from (batch x num_actions, *) to (batch, num_actions, *) """
        assert isinstance(self.action_space, Discrete), "action_space is non-discrete"
        assert isinstance(leaf, jnp.ndarray), f"leaf must be ndarray, got: {type(leaf)}"
        assert leaf.ndim >= 1, f"bad shape: {leaf.shape}"
        assert leaf.shape[0] % self.action_space.n == 0, \
            f"first axis size must be a multiple of num_actions, got shape: {leaf.shape}"
        return jnp.reshape(leaf, (-1, self.action_space.n, *leaf.shape[1:]))
        return leaf

    def _reshape_to_replicas(self, leaf):
        """ reshape from (batch, num_actions, *) to (batch x num_actions, *) """
        assert isinstance(self.action_space, Discrete), "action_space is non-discrete"
        assert isinstance(leaf, jnp.ndarray), f"leaf must be ndarray, got: {type(leaf)}"
        assert leaf.ndim >= 2, f"bad shape: {leaf.shape}"
        assert leaf.shape[1] == self.action_space.n, \
            f"axis=1 size must be num_actions, got shape: {leaf.shape}"
        return jnp.reshape(leaf, (-1, *leaf.shape[2:]))  # (batch x num_actions, *shape)
