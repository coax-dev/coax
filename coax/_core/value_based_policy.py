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

import gym
import jax
import jax.numpy as jnp
import haiku as hk

from ..utils import docstring
from ..proba_dists import CategoricalDist
from .base_policy import PolicyMixin
from .value_q import Q


__all__ = (
    'EpsilonGreedy',
    'BoltzmannPolicy',
)


class BaseValueBasedPolicy(PolicyMixin):
    """ Abstract base class for value-based policies. """

    def __init__(self, q):
        if not isinstance(q, Q):
            raise TypeError("q must be an instance of coax.Q")

        if not isinstance(q.action_space, gym.spaces.Discrete):
            raise TypeError(f"{self.__class__.__name__} is only well-defined for Discrete actions")

        self.q = q
        self.proba_dist = CategoricalDist(self.q.action_space)

    @property
    @docstring(Q.function)
    def function(self):
        return self._function

    @property
    @docstring(Q.function_state)
    def function_state(self):
        return self.q.function_state

    @function_state.setter
    def function_state(self, new_function_state):
        self.q.function_state = new_function_state

    @property
    def rng(self):
        return self.q.rng

    def _preprocess_state(self, s):
        return self.q._preprocess_state(s)


class EpsilonGreedy(BaseValueBasedPolicy):
    r"""

    Create an :math:`\epsilon`-greedy policy, given a q-function.

    This policy samples actions :math:`a\sim\pi(.|s)` according to the following rule:

    .. math::

        u &\sim \text{Uniform([0, 1])} \\
        a_\text{rand} &\sim \text{Uniform}(\text{actions}) \\
        a\ &=\ \left\{\begin{matrix}
            a_\text{rand} & \text{ if } u < \epsilon \\
            \arg\max_{a'} q(s,a') & \text{ otherwise }
        \end{matrix}\right.

    Parameters
    ----------
    q : Q

        A state-action value function.

    epsilon : float between 0 and 1, optional

        The probability of sampling an action uniformly at random (as opposed to sampling greedily).

    """
    def __init__(self, q, epsilon=0.1):
        super().__init__(q)
        self.epsilon = epsilon

        def func(params, state, rng, S, is_training):
            Q_s, new_state = self.q.function_type2(params['q'], state, rng, S, is_training)
            assert Q_s.ndim == 2
            assert Q_s.shape[1] == self.q.action_space.n
            A_greedy = (Q_s == Q_s.max(axis=1, keepdims=True)).astype(Q_s.dtype)
            A_greedy /= A_greedy.sum(axis=1, keepdims=True)  # there may be multiple max's (ties)
            A_greedy *= 1 - params['epsilon']
            A_greedy += params['epsilon'] / self.q.action_space.n
            dist_params = {'logits': jnp.log(A_greedy + 1e-15)}
            return dist_params, new_state

        self._function = jax.jit(func, static_argnums=(4,))

    @property
    @docstring(Q.params)
    def params(self):
        return hk.data_structures.to_immutable_dict({'epsilon': self.epsilon, 'q': self.q.params})

    @params.setter
    def params(self, new_params):
        if jax.tree_structure(new_params) != jax.tree_structure(self.params):
            raise TypeError("new params must have the same structure as old params")
        self.epsilon = new_params['epsilon']
        self.q.params = new_params['q']


class BoltzmannPolicy(BaseValueBasedPolicy):
    r"""

    Derive a Boltzmann policy from a q-function.

    This policy samples actions :math:`a\sim\pi(.|s)` according to the following rule:

    .. math::

        p &= \text{softmax}(q(s,.) / \tau) \\
        a &\sim \text{Cat}(p)

    Note that this policy is only well-defined for *discrete* action spaces.

    Parameters
    ----------
    q : Q

        A state-action value function.

    temperature : positive float, optional

        The Boltzmann temperature :math:`\tau>0` sets the sharpness of the categorical distribution.
        Picking a small value for :math:`\tau` results in greedy sampling while large values results
        in uniform sampling.

    """
    def __init__(self, q, temperature=0.02):
        super().__init__(q)
        self.temperature = temperature

        def func(params, state, rng, S, is_training):
            Q_s, new_state = self.q.function_type2(params['q'], state, rng, S, is_training)
            assert Q_s.ndim == 2
            assert Q_s.shape[1] == self.q.action_space.n
            dist_params = {'logits': Q_s / params['temperature']}
            return dist_params, new_state

        self._function = jax.jit(func, static_argnums=(4,))

    @property
    @docstring(Q.params)
    def params(self):
        return hk.data_structures.to_immutable_dict(
            {'temperature': self.temperature, 'q': self.q.params})

    @params.setter
    def params(self, new_params):
        if jax.tree_structure(new_params) != jax.tree_structure(self.params):
            raise TypeError("new params must have the same structure as old params")
        self.temperature = new_params['temperature']
        self.q.params = new_params['q']
