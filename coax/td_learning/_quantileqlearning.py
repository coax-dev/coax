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

import haiku as hk
import chex
from gym.spaces import Discrete
import jax

from ..utils import is_stochastic
from ._base import BaseTDLearningQuantileQWithTargetPolicy


class QuantileQLearning(BaseTDLearningQuantileQWithTargetPolicy):
    def __init__(self, q, pi_targ=None, q_targ=None,
                 optimizer=None, loss_function=None, policy_regularizer=None):
        super().__init__(
            q,
            pi_targ,
            q_targ=q_targ,
            optimizer=optimizer,
            loss_function=loss_function,
            policy_regularizer=policy_regularizer)

        # consistency checks
        if self.pi_targ is None and not isinstance(self.q.action_space, Discrete):
            raise TypeError("pi_targ must be provided if action space is not discrete")
        if self.pi_targ is not None and isinstance(self.q.action_space, Discrete):
            warnings.warn("pi_targ is ignored, because action space is discrete")

    def target_func(self, target_params, target_state, rng, transition_batch):
        rngs = hk.PRNGSequence(rng)

        if isinstance(self.q.action_space, Discrete):
            params, state = target_params['q_targ'], target_state['q_targ']
            S_next = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
            batch_size = jax.tree_leaves(S_next)[0].shape[0]
            quantiles = self.q.sample_quantiles(
                num_quantiles=self.q.num_quantiles, batch_size=batch_size, random_seed=next(rngs))

            Q_Quantiles_s, _ = self.q_targ.function_type4(
                params, state, next(rngs), S_next, quantiles, False)

            chex.assert_rank(Q_Quantiles_s, 3)
            assert Q_Quantiles_s.shape[1] == self.q_targ.action_space.n
            assert Q_Quantiles_s.shape[2] == self.q_targ.num_quantiles

            Q_s = Q_Quantiles_s.mean(axis=-1)

            # get greedy action as the argmax over q_targ
            A_next = (Q_s == Q_s.max(
                axis=1, keepdims=True)).astype(Q_s.dtype)
            A_next /= A_next.sum(axis=1, keepdims=True)  # there may be ties

        else:
            # get greedy action as the mode of pi_targ
            params, state = target_params['pi_targ'], target_state['pi_targ']
            S_next = self.pi_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
            A_next = self.pi_targ.mode_func(params, state, next(rngs), S_next)

        # evaluate on q (instead of q_targ)
        params, state = target_params['q'], target_state['q']
        S_next = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
        batch_size = jax.tree_leaves(S_next)[0].shape[0]
        quantiles = self.q.sample_quantiles(
            num_quantiles=self.q.num_quantiles, batch_size=batch_size, random_seed=next(rngs))

        Q_Quantiles_sa_next, _ = self.q.function_type3(
            params, state, next(rngs), S_next, A_next, quantiles, False)
        f, f_inv = self.q.value_transform.transform_func, self.q_targ.value_transform.inverse_func
        return f(transition_batch.Rn[..., None] + transition_batch.In[..., None]
                 * f_inv(Q_Quantiles_sa_next))
