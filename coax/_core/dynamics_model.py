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

import warnings
from inspect import signature

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from gym.spaces import Space, Discrete

from ..utils import safe_sample, single_to_batch, batch_to_single
from ..proba_dists import ProbaDist
from .base_func import BaseFunc, ExampleData, Inputs, ArgsType1, ArgsType2, ModelTypes


__all__ = (
    'DynamicsModel',
)


class DynamicsModel(BaseFunc):
    r"""

    A parametrized dynamics model :math:`p_\theta(s'|s,a)`. Here, :math:`s'` is the successor state,
    given that we take action :math:`a` from state :math:`s`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    observation_space : gym.Space

        The observation space of the environment. This is used to generate example input for
        initializing :attr:`params`.

    action_space : gym.Space

        The action space of the environment. This is used to generate example input for
        initializing :attr:`params`.

    action_preprocessor : function, optional

        Turns a single actions into a batch of actions that are compatible with the corresponding
        probability distribution. If left unspecified, this defaults to:

        .. code:: python

            action_preprocessor = ProbaDist(action_space).preprocess_variate

        See also :attr:`coax.proba_dists.ProbaDist.preprocess_variate`.

    proba_dist : ProbaDist, optional

        A probability distribution that is used to interpret the output of :code:`func
        <coax.Policy.func>`. Check out the :mod:`coax.proba_dists` module for available options.

        If left unspecified, this defaults to:

        .. code:: python

            proba_dist = coax.proba_dists.ProbaDist(observation_space)

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, observation_space, action_space,
            action_preprocessor=None, proba_dist=None, random_seed=None):

        self.proba_dist = ProbaDist(observation_space) if proba_dist is None else proba_dist
        self.action_preprocessor = \
            action_preprocessor if action_preprocessor is not None \
            else ProbaDist(action_space).preprocess_variate
        super().__init__(
            func=func,
            observation_space=observation_space,
            action_space=action_space,
            random_seed=random_seed)

    def __call__(self, s, a=None, return_logp=False):
        r"""

        Sample a successor state :math:`s'` from the dynamics model :math:`p(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        return_logp : bool, optional

            Whether to return the log-propensity :math:`\log p(s'|s,a)`.

        Returns
        -------
        s_next : state observation or list thereof

            Depending on whether :code:`a` is provided, this either returns a single next-state
            :math:`s'` or a list of :math:`n` next-states, one for each :math:`n` discrete actions.

        logp : non-positive float or list thereof, optional

            The log-propensity :math:`\log p(s'|s,a)`. This is only returned if we set
            ``return_logp=True``. Depending on whether :code:`a` is provided, this is either a
            single float or a list of :math:`n` floats, one for each :math:`n` discrete actions.

        """
        S = self.proba_dist.preprocess_variate(s)
        if a is None:
            # batch_to_single() projects: (batch, num_actions, *shape) -> (num_actions, *shape)
            S_next, logP = batch_to_single(
                self.sample_func_type2(self.params, self.function_state, self.rng, S))
            S_next = self.proba_dist.postprocess_variate(S_next, batch_mode=True)
            s_next = [batch_to_single(S_next, index=i) for i in range(self.action_space.n)]
            logp = list(logP)
        else:
            A = self.action_preprocessor(a)
            S_next, logP = self.sample_func_type1(self.params, self.function_state, self.rng, S, A)
            s_next = self.proba_dist.postprocess_variate(S_next)
            logp = batch_to_single(logP)
        return (s_next, logp) if return_logp else s_next

    def mode(self, s, a=None):
        r"""

        Get the most probable successor state :math:`s'` according to the dynamics model,
        :math:`s'=\arg\max_{s'}p(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        s_next : state observation or list thereof

            Depending on whether :code:`a` is provided, this either returns a single next-state
            :math:`s'` or a list of :math:`n` next-states, one for each :math:`n` discrete actions.

        """
        S = self.proba_dist.preprocess_variate(s)
        if a is None:
            # batch_to_single() projects: (batch, num_actions, *shape) -> (num_actions, *shape)
            S_next = batch_to_single(
                self.mode_func_type2(self.params, self.function_state, self.rng, S))
            S_next = self.proba_dist.postprocess_variate(S_next, batch_mode=True)
            s_next = [batch_to_single(S_next, index=i) for i in range(self.action_space.n)]
        else:
            A = self.action_preprocessor(a)
            S_next = self.mode_func_type1(self.params, self.function_state, self.rng, S, A)
            s_next = self.proba_dist.postprocess_variate(S_next)
        return s_next

    def dist_params(self, s, a=None):
        r"""

        Get the parameters of the conditional probability distribution :math:`p(s'|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        dist_params : dict or list of dicts

            Depending on whether :code:`a` is provided, this either returns a single dist-params
            dict or a list of :math:`n` such dicts, one for each :math:`n` discrete actions.

        """
        S = self.proba_dist.preprocess_variate(s)
        if a is None:
            # batch_to_single() projects: (batch, num_actions, *shape) -> (num_actions, *shape)
            dist_params = batch_to_single(
                self.function_type2(self.params, self.function_state, self.rng, S, False)[0])
            dist_params = [
                batch_to_single(dist_params, index=i) for i in range(self.action_space.n)]
        else:
            A = self.action_preprocessor(a)
            dist_params, _ = self.function_type1(
                self.params, self.function_state, self.rng, S, A, False)
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
            # example: let S = [7, 2, 5, 8] and num_actions = 3, then
            # S_rep = [7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8]  # repeated
            # A_rep = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # tiled
            S_rep = jax.tree_map(lambda x: jnp.repeat(x, n, axis=0), S)
            A_rep = jnp.tile(jnp.arange(n), S.shape[0])
            A_rep = self.action_preprocessor(A_rep)  # one-hot encoding

            # evaluate on replicas => output shape: (batch * num_actions, *shape)
            dist_params_rep, state_new = self.function(
                type1_params, type1_state, rng, S_rep, A_rep, is_training)
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
            cls, observation_space, action_space,
            action_preprocessor=None, proba_dist=None, batch_size=1, random_seed=None):

        if not isinstance(observation_space, Space):
            raise TypeError(
                f"observation_space must be derived from gym.Space, got: {type(observation_space)}")

        rnd = onp.random.RandomState(random_seed)

        if proba_dist is None:
            proba_dist = ProbaDist(observation_space)

        # input: state observations
        S = [safe_sample(observation_space, rnd) for _ in range(batch_size)]
        S = [proba_dist.preprocess_variate(s) for s in S]
        S = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *S)

        # input: actions
        A = jax.tree_multimap(
            lambda *x: jnp.stack(x, axis=0),
            *(safe_sample(action_space, rnd) for _ in range(batch_size)))
        try:
            if action_preprocessor is None:
                action_preprocessor = ProbaDist(action_space).preprocess_variate
            A = action_preprocessor(A)
        except Exception as e:
            warnings.warn(f"preprocessing failed for actions A; caught exception: {e}")

        # output
        dist_params_type1 = jax.tree_map(
            lambda x: jnp.asarray(rnd.randn(batch_size, *x.shape[1:])), proba_dist.default_priors)

        q1_data = ExampleData(
            inputs=Inputs(args=ArgsType1(S=S, A=A, is_training=True), static_argnums=(2,)),
            output=dist_params_type1)
        q2_data = None
        if isinstance(action_space, Discrete):
            dist_params_type2 = jax.tree_map(
                lambda x: jnp.asarray(rnd.randn(batch_size, action_space.n, *x.shape[1:])),
                proba_dist.default_priors)
            q2_data = ExampleData(
                inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
                output=dist_params_type2)

        return ModelTypes(type1=q1_data, type2=q2_data)

    @property
    def sample_func_type1(self):
        r"""

        The function that is used for sampling *random* actions, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.sample_func_type1(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_sample_func_type1'):
            def func(params, state, rng, S, A):
                rngs = hk.PRNGSequence(rng)
                dist_params, _ = self.function_type1(params, state, next(rngs), S, A, False)
                S_next = self.proba_dist.sample(dist_params, next(rngs))
                logP = self.proba_dist.log_proba(dist_params, S_next)
                return S_next, logP
            self._sample_func_type1 = jax.jit(func)
        return self._sample_func_type1

    @property
    def sample_func_type2(self):
        r"""

        The function that is used for sampling *random* actions, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.sample_func_type2(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_sample_func_type2'):
            def func(params, state, rng, S):
                rngs = hk.PRNGSequence(rng)
                dist_params, _ = self.function_type2(params, state, next(rngs), S, False)
                dist_params = jax.tree_map(self._reshape_to_replicas, dist_params)
                S_next = self.proba_dist.sample(dist_params, next(rngs))    # (batch x n, *shape)
                logP = self.proba_dist.log_proba(dist_params, S_next)       # (batch x n)
                S_next = jax.tree_map(self._reshape_from_replicas, S_next)  # (batch, n, *shape)
                logP = self._reshape_from_replicas(logP)                    # (batch, n)
                return S_next, logP
            self._sample_func_type2 = jax.jit(func)
        return self._sample_func_type2

    @property
    def mode_func_type1(self):
        r"""

        The function that is used for sampling *greedy* actions, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.mode_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_mode_func_type1'):
            def func(params, state, rng, S, A):
                dist_params, _ = self.function_type1(params, state, rng, S, A, False)
                S_next = self.proba_dist.mode(dist_params)
                return S_next
            self._mode_func_type1 = jax.jit(func)
        return self._mode_func_type1

    @property
    def mode_func_type2(self):
        r"""

        The function that is used for sampling *greedy* actions, defined as a JIT-compiled pure
        function. This function may be called directly as:

        .. code:: python

            output = obj.mode_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        if not hasattr(self, '_mode_func_type2'):
            def func(params, state, rng, S):
                dist_params, _ = self.function_type2(params, state, rng, S, False)
                dist_params = jax.tree_map(self._reshape_to_replicas, dist_params)
                S_next = self.proba_dist.mode(dist_params)                  # (batch x n, *shape)
                S_next = jax.tree_map(self._reshape_from_replicas, S_next)  # (batch, n, *shape)
                return S_next
            self._mode_func_type2 = jax.jit(func)
        return self._mode_func_type2

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
                "type-2 dynamics models are only well-defined for Discrete action spaces")

        example_data_per_modeltype = self.example_data(
            observation_space=self.observation_space,
            action_space=self.action_space,
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
        """ reshape from (batch x num_actions, *shape) to (batch, *shape, num_actions) """
        assert isinstance(self.action_space, Discrete), "action_space is non-discrete"
        assert isinstance(leaf, jnp.ndarray), f"leaf must be ndarray, got: {type(leaf)}"
        assert leaf.ndim >= 1, f"bad shape: {leaf.shape}"
        assert leaf.shape[0] % self.action_space.n == 0, \
            f"first axis size must be a multiple of num_actions, got shape: {leaf.shape}"
        return jnp.reshape(leaf, (-1, self.action_space.n, *leaf.shape[1:]))
        return leaf

    def _reshape_to_replicas(self, leaf):
        """ reshape from (batch, *shape, num_actions) to (batch x num_actions, *shape) """
        assert isinstance(self.action_space, Discrete), "action_space is non-discrete"
        assert isinstance(leaf, jnp.ndarray), f"leaf must be ndarray, got: {type(leaf)}"
        assert leaf.ndim >= 2, f"bad shape: {leaf.shape}"
        assert leaf.shape[1] == self.action_space.n, \
            f"axis=1 size must be num_actions, got shape: {leaf.shape}"
        return jnp.reshape(leaf, (-1, *leaf.shape[2:]))  # (batch x num_actions, *shape)
