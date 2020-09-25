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

import jax.numpy as jnp
import haiku as hk
from gym.spaces import Discrete

from ._base import BaseTDLearningQWithTargetPolicy


class QLearning(BaseTDLearningQWithTargetPolicy):
    r"""

    TD-learning with Q-Learning updates.

    The :math:`n`-step bootstrapped target for discrete actions is constructed as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,\max_aq_\text{targ}\left(S_{t+n}, a\right)

    For non-discrete action spaces, this uses a DDPG-style target:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,q_\text{targ}\left(
            S_{t+n}, a_\text{targ}(S_{t+n})\right)

    where :math:`a_\text{targ}(s)` is the **mode** of the underlying conditional probability
    distribution :math:`\pi_\text{targ}(.|s)`. Even though these two formulations of the q-learning
    target are equivalent, the implementation of the latter does require additional input, namely
    :code:`pi_targ`.

    The :math:`n`-step reward and indicator (referenced above) are defined as:

    .. math::

        R^{(n)}_t\ &=\ \sum_{k=0}^{n-1}\gamma^kR_{t+k} \\
        I^{(n)}_t\ &=\ \left\{\begin{matrix}
            0           & \text{if $S_{t+n}$ is a terminal state} \\
            \gamma^n    & \text{otherwise}
        \end{matrix}\right.


    Parameters
    ----------
    q : Q

        The main q-function to update.

    pi_targ : Policy, optional

        The policy that is used for constructing the TD-target. This is ignored if the action space
        is discrete and *required* otherwise.

    q_targ : Q, optional

        The q-function that is used for constructing the TD-target. If this is left unspecified, we
        set ``q_targ = q`` internally.

    optimizer : optax optimizer, optional

        An optax-style optimizer. The default optimizer is :func:`optax.adam(1e-3)
        <optax.adam>`.

    loss_function : callable, optional

        The loss function that will be used to regress to the (bootstrapped) target. The loss
        function is expected to be of the form:

        .. math::

            L(y_\text{true}, y_\text{pred})\in\mathbb{R}

        If left unspecified, this defaults to :func:`coax.value_losses.huber`. Check out the
        :mod:`coax.value_losses` module for other predefined loss functions.

    policy_regularizer : Regularizer, optional

        If provided, this policy regularizer is added to the TD-target. A typical example is to use
        an :class:`coax.regularizers.EntropyRegularizer`, which adds the policy entropy to
        the target. In this case, we minimize the following loss shifted by the entropy term:

        .. math::

            L(y_\text{true} + \beta\,H[\pi], y_\text{pred})

        Note that the coefficient :math:`\beta` plays the role of the temperature in SAC-style
        agents.

    """
    def __init__(
            self, q, pi_targ=None, q_targ=None,
            optimizer=None, loss_function=None, policy_regularizer=None):

        super().__init__(
            q=q,
            pi_targ=pi_targ,
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
            # get greedy value directly from q_targ
            params, state = target_params['q_targ'], target_state['q_targ']
            S_next = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
            Q_s, _ = self.q_targ.function_type2(params, state, next(rngs), S_next, False)
            assert Q_s.ndim == 2, f"bad shape: {Q_s.shape}"
            Q_sa = jnp.max(Q_s, axis=1)

        else:
            # get greedy action from pi_targ
            params, state = target_params['pi_targ'], target_state['pi_targ']
            S_next = self.pi_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
            A_next = self.pi_targ.mode_func(params, state, next(rngs), S_next)

            # evaluate q_targ on greedy action
            params, state = target_params['q_targ'], target_state['q_targ']
            S_next = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
            Q_sa, _ = self.q_targ.function_type1(params, state, next(rngs), S_next, A_next, False)

        assert Q_sa.ndim == 1, f"bad shape: {Q_sa.shape}"
        f, f_inv = self.q.value_transform
        return f(transition_batch.Rn + transition_batch.In * f_inv(Q_sa))
