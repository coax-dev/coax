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

import jax
import haiku as hk

from .._core.v import V
from .._core.q import Q
from .._core.stochastic_transition_model import StochasticTransitionModel
from .._core.stochastic_reward_function import StochasticRewardFunction


__all__ = (
    'SuccessorStateQ',
)


class SuccessorStateQ:
    r"""

    A state-action value function :math:`q(s,a)=r(s,a)+\gamma\mathop{\mathbb{E}}_{s'\sim
    p(.|s,a)}v(s')`.

    Parameters
    ----------
    v : V

        A state value function :math:`v(s)`.

    p : StochasticTransitionModel

        A dynamics model :math:`p(s'|s,a)`. This may also be a ordinary function with the signature:
        :code:`(Observation, Action) -> Observation`.

    r : StochasticRewardFunction

        A reward function :math:`r(s,a)`. This may also be a ordinary function with the signature:
        :code:`(Observation, Action) -> float`.

    gamma : float between 0 and 1, optional

        The discount factor for future rewards :math:`\gamma\in[0,1]`.

    """
    def __init__(self, v, p, r, gamma=0.9):
        # some explicit type checks
        if not isinstance(v, V):
            raise TypeError(f"v must be of type V, got: {type(v)}")
        if not isinstance(p, StochasticTransitionModel):
            raise TypeError(f"p must be of type StochasticTransitionModel, got: {type(p)}")
        if not isinstance(r, StochasticRewardFunction):
            raise TypeError(f"r must be of type StochasticRewardFunction, got: {type(r)}")

        self.v = v
        self.p = p
        self.r = r
        self.gamma = gamma

        # we assume that self.r uses the same action preprocessor
        self.observation_space = self.p.observation_space
        self.action_space = self.p.action_space
        self.action_preprocessor = self.p.action_preprocessor
        self.observation_preprocessor = self.p.observation_preprocessor

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
                S_ = self.p.observation_preprocessor(next(rngs), S)
                dist_params, new_state['p'] = \
                    self.p.function_type1(params['p'], state['p'], next(rngs), S_, A, is_training)
                S_next = self.p.proba_dist.sample(dist_params, next(rngs))
                S_next = self.p.proba_dist.postprocess_variate(next(rngs), S_next, batch_mode=True)

                # r = r(s,a)
                S_ = self.r.observation_preprocessor(next(rngs), S)
                dist_params, new_state['r'] = \
                    self.r.function_type1(params['r'], state['r'], next(rngs), S_, A, is_training)
                R = self.r.proba_dist.sample(dist_params, next(rngs))
                R = self.r.proba_dist.postprocess_variate(next(rngs), R, batch_mode=True)

                # v(s')
                V, new_state['v'] = \
                    self.v.function(params['v'], state['v'], next(rngs), S_next, is_training)

                # q = r + γ v(s')
                Q_sa = R + params['gamma'] * V
                assert Q_sa.ndim == 1, f"bad shape: {Q_sa.shape}"

                return Q_sa, hk.data_structures.to_immutable_dict(new_state)

            self._function_type1 = jax.jit(func, static_argnums=(5,))

        return self._function_type1

    @property
    def function_type2(self):
        if not hasattr(self, '_function_type2'):
            def func(params, state, rng, S, is_training):
                rngs = hk.PRNGSequence(rng)
                new_state = dict(state)

                # s' ~ p(s'|s,.)  # note: S_next is replicated, one for each (discrete) action
                S_ = self.p.observation_preprocessor(next(rngs), S)
                dist_params_rep, new_state['p'] = \
                    self.p.function_type2(params['p'], state['p'], next(rngs), S_, is_training)
                dist_params_rep = jax.tree_map(self.p._reshape_to_replicas, dist_params_rep)
                S_next_rep = self.p.proba_dist.sample(dist_params_rep, next(rngs))

                # r ~ p(r|s,a)  # note: R is replicated, one for each (discrete) action
                S_ = self.r.observation_preprocessor(next(rngs), S)
                dist_params_rep, new_state['r'] = \
                    self.r.function_type2(params['r'], state['r'], next(rngs), S_, is_training)
                dist_params_rep = jax.tree_map(self.r._reshape_to_replicas, dist_params_rep)
                R_rep = self.r.proba_dist.sample(dist_params_rep, next(rngs))
                R_rep = self.r.proba_dist.postprocess_variate(next(rngs), R_rep, batch_mode=True)

                # v(s')  # note: since the input S_next is replicated, so is the output V
                S_next_rep = \
                    self.p.proba_dist.postprocess_variate(next(rngs), S_next_rep, batch_mode=True)
                V_rep, new_state['v'] = \
                    self.v.function(params['v'], state['v'], next(rngs), S_next_rep, is_training)

                # q = r + γ v(s')
                Q_rep = R_rep + params['gamma'] * V_rep

                # reshape from (batch x num_actions, *) to (batch, num_actions, *)
                Q_s = self.p._reshape_from_replicas(Q_rep)
                assert Q_s.ndim == 2, f"bad shape: {Q_s.shape}"
                assert Q_s.shape[1] == self.action_space.n, f"bad shape: {Q_s.shape}"

                return Q_s, hk.data_structures.to_immutable_dict(new_state)

            self._function_type2 = jax.jit(func, static_argnums=(4,))

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
