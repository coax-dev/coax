from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from gym.spaces import Space, Discrete

from ..utils import safe_sample, batch_to_single, default_preprocessor
from ..proba_dists import ProbaDist
from .base_func import BaseFunc, ExampleData, Inputs, ArgsType1, ArgsType2, ModelTypes


__all__ = (
    'TransitionModel',
)


class TransitionModel(BaseFunc):
    r"""

    A deterministic transition function :math:`s'_\theta(s,a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass. The function signature must be the
        same as the example below.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations in a form that is convenient for
        feeding into :code:`func`. If left unspecified, this defaults to
        :attr:`proba_dist.preprocess_variate <coax.proba_dists.ProbaDist.preprocess_variate>`. The
        reason why the default is not :func:`coax.utils.default_preprocessor` is that we prefer
        consistence with :class:`coax.StochasticTransitionModel`.

    observation_postprocessor : function, optional

        Takes a batch of generated observations and makes sure that they are that are compatible
        with the original :code:`observation_space`. If left unspecified, this defaults to
        :attr:`proba_dist.postprocess_variate <coax.proba_dists.ProbaDist.postprocess_variate>`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions in a form that is convenient for feeding into
        :code:`func`. If left unspecified, this defaults
        :func:`default_preprocessor(env.action_space) <coax.utils.default_preprocessor>`.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, env, observation_preprocessor=None, observation_postprocessor=None,
            action_preprocessor=None, random_seed=None):

        self.observation_preprocessor = observation_preprocessor
        self.observation_postprocessor = observation_postprocessor
        self.action_preprocessor = action_preprocessor

        # defaults
        if self.observation_preprocessor is None:
            self.observation_preprocessor = ProbaDist(env.observation_space).preprocess_variate
        if self.observation_postprocessor is None:
            self.observation_postprocessor = ProbaDist(env.observation_space).postprocess_variate
        if self.action_preprocessor is None:
            self.action_preprocessor = default_preprocessor(env.action_space)

        super().__init__(
            func,
            observation_space=env.observation_space,
            action_space=env.action_space,
            random_seed=random_seed)

    def __call__(self, s, a=None):
        r"""

        Evaluate the state-action function on a state observation :math:`s` or
        on a state-action pair :math:`(s, a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action

            A single action :math:`a`.

        Returns
        -------
        q_sa or q_s : ndarray

            Depending on whether :code:`a` is provided, this either returns a scalar representing
            :math:`q(s,a)\in\mathbb{R}` or a vector representing :math:`q(s,.)\in\mathbb{R}^n`,
            where :math:`n` is the number of discrete actions. Naturally, this only applies for
            discrete action spaces.

        """
        S = self.observation_preprocessor(self.rng, s)
        if a is None:
            S_next, _ = self.function_type2(self.params, self.function_state, self.rng, S, False)
            S_next = batch_to_single(S_next)  # (batch, num_actions, *) -> (num_actions, *)
            n = self.action_space.n
            s_next = [self.observation_postprocessor(self.rng, S_next, index=i) for i in range(n)]
        else:
            A = self.action_preprocessor(self.rng, a)
            S_next, _ = self.function_type1(self.params, self.function_state, self.rng, S, A, False)
            s_next = self.observation_postprocessor(self.rng, S_next)
        return s_next

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
            S_next, state_new = self.function(type2_params, type2_state, rng, S, is_training)
            S_next = jax.tree_map(project(A), S_next)
            return S_next, state_new

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
            S_next_rep, state_new = self.function(
                type1_params, type1_state, next(rngs), S_rep, A_rep, is_training)
            S_next = jax.tree_map(reshape, S_next_rep)

            return S_next, state_new

        return type2_func

    @property
    def modeltype(self):
        r"""

        Specifier for how the transition function is modeled, i.e.

        .. math::

            (s,a)   &\mapsto s'(s,a)    &\qquad (\text{modeltype} &= 1) \\
            s       &\mapsto s'(s,.)    &\qquad (\text{modeltype} &= 2)

        Note that modeltype=2 is only well-defined if the action space is :class:`Discrete
        <gym.spaces.Discrete>`. Namely, :math:`n` is the number of discrete actions.

        """
        return self._modeltype

    @classmethod
    def example_data(
            cls, env, observation_preprocessor=None, action_preprocessor=None,
            batch_size=1, random_seed=None):

        if not isinstance(env.observation_space, Space):
            raise TypeError(
                "env.observation_space must be derived from gym.Space, "
                f"got: {type(env.observation_space)}")

        if not isinstance(env.action_space, Space):
            raise TypeError(
                f"env.action_space must be derived from gym.Space, got: {type(env.action_space)}")

        if observation_preprocessor is None:
            observation_preprocessor = ProbaDist(env.observation_space).preprocess_variate

        if action_preprocessor is None:
            action_preprocessor = default_preprocessor(env.action_space)

        rnd = onp.random.RandomState(random_seed)
        rngs = hk.PRNGSequence(rnd.randint(jnp.iinfo('int32').max))

        # input: state observations
        S = [safe_sample(env.observation_space, rnd) for _ in range(batch_size)]
        S = [observation_preprocessor(next(rngs), s) for s in S]
        S = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *S)

        # input: actions
        A = [safe_sample(env.action_space, rnd) for _ in range(batch_size)]
        A = [action_preprocessor(next(rngs), a) for a in A]
        A = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *A)

        # output: type1
        S_next_type1 = jax.tree_map(lambda x: jnp.asarray(rnd.randn(batch_size, *x.shape[1:])), S)
        q1_data = ExampleData(
            inputs=Inputs(args=ArgsType1(S=S, A=A, is_training=True), static_argnums=(2,)),
            output=S_next_type1)

        if not isinstance(env.action_space, Discrete):
            return ModelTypes(type1=q1_data, type2=None)

        # output: type2 (if actions are discrete)
        S_next_type2 = jax.tree_map(
            lambda x: jnp.asarray(rnd.randn(batch_size, env.action_space.n, *x.shape[1:])), S)
        q2_data = ExampleData(
            inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
            output=S_next_type2)

        return ModelTypes(type1=q1_data, type2=q2_data)

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
            raise TypeError("type-2 models are only well-defined for Discrete action spaces")

        Env = namedtuple('Env', ('observation_space', 'action_space'))
        example_data_per_modeltype = self.example_data(
            env=Env(self.observation_space, self.action_space),
            action_preprocessor=self.action_preprocessor,
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
