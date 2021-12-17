import warnings

import jax
import haiku as hk

from .._core.q import Q
from .._core.base_stochastic_func_type1 import BaseStochasticFuncType1
from ..utils import (
    check_preprocessors, is_vfunction, is_reward_function, is_transition_model, is_stochastic, jit)


__all__ = (
    'SuccessorStateQ',
)


class SuccessorStateQ:
    r"""

    A state-action value function :math:`q(s,a)=r(s,a)+\gamma\mathop{\mathbb{E}}_{s'\sim
    p(.|s,a)}v(s')`.

    **caution** A word of caution: If you use custom observation/action pre-/post-processors, please
    make sure that all three function approximators :code:`v`, :code:`p` and :code:`r` use the same
    ones.

    Parameters
    ----------
    v : V or StochasticV

        A state value function :math:`v(s)`.

    p : TransitionModel or StochasticTransitionModel

        A transition model.

    r : RewardFunction or StochasticRewardFunction

        A reward function.

    gamma : float between 0 and 1, optional

        The discount factor for future rewards :math:`\gamma\in[0,1]`.

    """
    def __init__(self, v, p, r, gamma=0.9):
        # some explicit type checks
        if not is_vfunction(v):
            raise TypeError(f"v must be a state-value function, got: {type(v)}")
        if not is_transition_model(p):
            raise TypeError(f"p must be a transition model, got: {type(p)}")
        if not is_reward_function(r):
            raise TypeError(f"r must be a reward function, got: {type(r)}")

        self.v = v
        self.p = p
        self.r = r
        self.gamma = gamma

        # we assume that self.r uses the same action preprocessor
        self.observation_space = self.p.observation_space
        self.action_space = self.p.action_space
        self.action_preprocessor = self.p.action_preprocessor
        self.observation_preprocessor = self.p.observation_preprocessor
        self.observation_postprocessor = self.p.observation_postprocessor
        self.value_transform = self.v.value_transform

        if not check_preprocessors(
                self.observation_space,
                self.v.observation_preprocessor,
                self.r.observation_preprocessor,
                self.p.observation_preprocessor):
            warnings.warn(
                "it seems that observation_preprocessors of v, r ,p do not match; please "
                "instantiate your functions approximators with the same observation_preprocessors, "
                "e.g. v = coax.V(..., observation_preprocessor=p.observation_preprocessor) and "
                "r = coax.RewardFunction(..., observation_preprocessor=p.observation_preprocessor) "
                "to ensure that all preprocessors match")

    _reshape_to_replicas = BaseStochasticFuncType1._reshape_to_replicas
    _reshape_from_replicas = BaseStochasticFuncType1._reshape_from_replicas

    @property
    def rng(self):
        return self.v.rng

    @property
    def params(self):
        return hk.data_structures.to_immutable_dict({
            'v': self.v.params,
            'p': self.p.params,
            'r': self.r.params,
            'gamma': self.gamma,
        })

    @property
    def function_state(self):
        return hk.data_structures.to_immutable_dict({
            'v': self.v.function_state,
            'p': self.p.function_state,
            'r': self.r.function_state,
        })

    @property
    def function_type1(self):
        if not hasattr(self, '_function_type1'):
            def func(params, state, rng, S, A, is_training):
                rngs = hk.PRNGSequence(rng)
                new_state = dict(state)

                # s' ~ p(.|s,a)
                if is_stochastic(self.p):
                    dist_params, new_state['p'] = self.p.function_type1(
                        params['p'], state['p'], next(rngs), S, A, is_training)
                    S_next = self.p.proba_dist.mean(dist_params)
                else:
                    S_next, new_state['p'] = self.p.function_type1(
                        params['p'], state['p'], next(rngs), S, A, is_training)

                # r = r(s,a)
                if is_stochastic(self.r):
                    dist_params, new_state['r'] = self.r.function_type1(
                        params['r'], state['r'], next(rngs), S, A, is_training)
                    R = self.r.proba_dist.mean(dist_params)
                    R = self.r.proba_dist.postprocess_variate(next(rngs), R, batch_mode=True)
                else:
                    R, new_state['r'] = self.r.function_type1(
                        params['r'], state['r'], next(rngs), S, A, is_training)

                # v(s')
                if is_stochastic(self.v):
                    dist_params, new_state['v'] = self.v.function(
                        params['v'], state['v'], next(rngs), S_next, is_training)
                    V = self.v.proba_dist.mean(dist_params)
                    V = self.v.proba_dist.postprocess_variate(next(rngs), V, batch_mode=True)
                else:
                    V, new_state['v'] = self.v.function(
                        params['v'], state['v'], next(rngs), S_next, is_training)

                # q = r + γ v(s')
                f, f_inv = self.value_transform
                Q_sa = f(R + params['gamma'] * f_inv(V))
                assert Q_sa.ndim == 1, f"bad shape: {Q_sa.shape}"

                new_state = hk.data_structures.to_immutable_dict(new_state)
                assert jax.tree_structure(new_state) == jax.tree_structure(state)

                return Q_sa, new_state

            self._function_type1 = jit(func, static_argnums=(5,))

        return self._function_type1

    @property
    def function_type2(self):
        if not hasattr(self, '_function_type2'):
            def func(params, state, rng, S, is_training):
                rngs = hk.PRNGSequence(rng)
                new_state = dict(state)

                # s' ~ p(s'|s,.)  # note: S_next is replicated, one for each (discrete) action
                if is_stochastic(self.p):
                    dist_params_rep, new_state['p'] = self.p.function_type2(
                        params['p'], state['p'], next(rngs), S, is_training)
                    dist_params_rep = jax.tree_map(self._reshape_to_replicas, dist_params_rep)
                    S_next_rep = self.p.proba_dist.mean(dist_params_rep)
                else:
                    S_next_rep, new_state['p'] = self.p.function_type2(
                        params['p'], state['p'], next(rngs), S, is_training)
                    S_next_rep = jax.tree_map(self._reshape_to_replicas, S_next_rep)

                # r ~ p(r|s,a)  # note: R is replicated, one for each (discrete) action
                if is_stochastic(self.r):
                    dist_params_rep, new_state['r'] = self.r.function_type2(
                        params['r'], state['r'], next(rngs), S, is_training)
                    dist_params_rep = jax.tree_map(self._reshape_to_replicas, dist_params_rep)
                    R_rep = self.r.proba_dist.mean(dist_params_rep)
                    R_rep = self.r.proba_dist.postprocess_variate(
                        next(rngs), R_rep, batch_mode=True)
                else:
                    R_rep, new_state['r'] = self.r.function_type2(
                        params['r'], state['r'], next(rngs), S, is_training)
                    R_rep = jax.tree_map(self._reshape_to_replicas, R_rep)

                # v(s')  # note: since the input S_next is replicated, so is the output V
                if is_stochastic(self.v):
                    dist_params_rep, new_state['v'] = self.v.function(
                        params['v'], state['v'], next(rngs), S_next_rep, is_training)
                    V_rep = self.v.proba_dist.mean(dist_params_rep)
                    V_rep = self.v.proba_dist.postprocess_variate(
                        next(rngs), V_rep, batch_mode=True)
                else:
                    V_rep, new_state['v'] = self.v.function(
                        params['v'], state['v'], next(rngs), S_next_rep, is_training)

                # q = r + γ v(s')
                f, f_inv = self.value_transform
                Q_rep = f(R_rep + params['gamma'] * f_inv(V_rep))

                # reshape from (batch x num_actions, *) to (batch, num_actions, *)
                Q_s = self._reshape_from_replicas(Q_rep)
                assert Q_s.ndim == 2, f"bad shape: {Q_s.shape}"
                assert Q_s.shape[1] == self.action_space.n, f"bad shape: {Q_s.shape}"

                new_state = hk.data_structures.to_immutable_dict(new_state)
                assert jax.tree_structure(new_state) == jax.tree_structure(state)

                return Q_s, new_state

            self._function_type2 = jit(func, static_argnums=(4,))

        return self._function_type2

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
        return Q.__call__(self, s, a=a)
