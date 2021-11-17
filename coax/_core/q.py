from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from gym.spaces import Space, Discrete

from ..utils import safe_sample, default_preprocessor
from ..value_transforms import ValueTransform
from .base_func import BaseFunc, ExampleData, Inputs, ArgsType1, ArgsType2, ModelTypes


__all__ = (
    'Q',
)


class Q(BaseFunc):
    r"""

    A state-action value function :math:`q_\theta(s,a)`.

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
        :func:`default_preprocessor(env.observation_space) <coax.utils.default_preprocessor>`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions in a form that is convenient for feeding into
        :code:`func`. If left unspecified, this defaults
        :func:`default_preprocessor(env.action_space) <coax.utils.default_preprocessor>`.

    value_transform : ValueTransform or pair of funcs, optional

        If provided, the target for the underlying function approximator is transformed such that:

        .. math::

            \tilde{q}_\theta(S_t, A_t)\ \approx\ f(G_t)

        This means that calling the function involves undoing this transformation:

        .. math::

            q(s, a)\ =\ f^{-1}(\tilde{q}_\theta(s, a))

        Here, :math:`f` and :math:`f^{-1}` are given by ``value_transform.transform_func`` and
        ``value_transform.inverse_func``, respectively. Note that a ValueTransform is just a
        glorified pair of functions, i.e. passing ``value_transform=(func, inverse_func)`` works
        just as well.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, env, observation_preprocessor=None, action_preprocessor=None,
            value_transform=None, random_seed=None):

        self.observation_preprocessor = observation_preprocessor
        self.action_preprocessor = action_preprocessor
        self.value_transform = value_transform

        # defaults
        if self.observation_preprocessor is None:
            self.observation_preprocessor = default_preprocessor(env.observation_space)
        if self.action_preprocessor is None:
            self.action_preprocessor = default_preprocessor(env.action_space)
        if self.value_transform is None:
            self.value_transform = ValueTransform(lambda x: x, lambda x: x)
        if not isinstance(self.value_transform, ValueTransform):
            self.value_transform = ValueTransform(*value_transform)

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
            Q, _ = self.function_type2(self.params, self.function_state, self.rng, S, False)
        else:
            A = self.action_preprocessor(self.rng, a)
            Q, _ = self.function_type1(self.params, self.function_state, self.rng, S, A, False)
        Q = self.value_transform.inverse_func(Q)
        return onp.asarray(Q[0])

    @property
    def function_type1(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-1 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 1:
            return self.function

        assert isinstance(self.action_space, Discrete)

        def q1_func(q2_params, q2_state, rng, S, A, is_training):
            assert A.ndim == 2
            assert A.shape[1] == self.action_space.n
            Q_s, state_new = self.function(q2_params, q2_state, rng, S, is_training)
            Q_sa = jax.vmap(jnp.dot)(A, Q_s)
            return Q_sa, state_new

        return q1_func

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
                "input 'A' is required for type-1 q-function when action space is non-Discrete")

        n = self.action_space.n

        def q2_func(q1_params, q1_state, rng, S, is_training):
            rngs = hk.PRNGSequence(rng)
            batch_size = jax.tree_leaves(S)[0].shape[0]

            # example: let S = [7, 2, 5, 8] and num_actions = 3, then
            # S_rep = [7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8]  # repeated
            # A_rep = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # tiled
            S_rep = jax.tree_map(lambda x: jnp.repeat(x, n, axis=0), S)
            A_rep = jnp.tile(jnp.arange(n), batch_size)
            A_rep = self.action_preprocessor(next(rngs), A_rep)  # one-hot encoding

            # evaluate on replicas => output shape: (batch * num_actions, 1)
            Q_sa_rep, state_new = \
                self.function(q1_params, q1_state, next(rngs), S_rep, A_rep, is_training)
            Q_s = Q_sa_rep.reshape(-1, n)  # shape: (batch, num_actions)

            return Q_s, state_new

        return q2_func

    @property
    def modeltype(self):
        r"""

        Specifier for how the q-function is modeled, i.e.

        .. math::

            (s,a)   &\mapsto q(s,a)\in\mathbb{R}    &\qquad (\text{modeltype} &= 1) \\
            s       &\mapsto q(s,.)\in\mathbb{R}^n  &\qquad (\text{modeltype} &= 2)

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

        if observation_preprocessor is None:
            observation_preprocessor = default_preprocessor(env.observation_space)

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
        q1_data = ExampleData(
            inputs=Inputs(args=ArgsType1(S=S, A=A, is_training=True), static_argnums=(2,)),
            output=jnp.asarray(rnd.randn(batch_size)),
        )

        if not isinstance(env.action_space, Discrete):
            return ModelTypes(type1=q1_data, type2=None)

        # output: type2 (if actions are discrete)
        q2_data = ExampleData(
            inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
            output=jnp.asarray(rnd.randn(batch_size, env.action_space.n)),
        )

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
            raise TypeError("type-2 q-functions are only well-defined for Discrete action spaces")

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
        if not isinstance(actual, jnp.ndarray):
            class_name = actual.__class__.__name__
            raise TypeError(f"func has bad return type; expected jnp.ndarray, got {class_name}")

        if not jnp.issubdtype(actual.dtype, jnp.floating):
            raise TypeError(
                "func has bad return dtype; expected a subdtype of jnp.floating, "
                f"got dtype={actual.dtype}")

        if actual.shape != expected.shape:
            raise TypeError(
                f"func has bad return shape, expected: {expected.shape}, got: {actual.shape}")
