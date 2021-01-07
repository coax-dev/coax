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

import jax
import jax.numpy as jnp
import haiku as hk
import chex
from gym.spaces import Discrete

from .._core.q import Q
from ..utils import get_grads_diagnostics, is_policy, is_stochastic, jit
from ._base import BaseTDLearning


class ClippedDoubleQLearning(BaseTDLearning):  # TODO(krholshe): make this less ugly
    r"""

    TD-learning with `TD3 <https://arxiv.org/abs/1802.09477>`_ style double q-learning updates, in
    which the target network is only used in selecting the would-be next action.

    For discrete actions, the :math:`n`-step bootstrapped target is constructed as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,\min_{i,j}q_i(S_{t+n}, \arg\max_a q_j(S_{t+n}, a))

    where :math:`q_i(s,a)` is the :math:`i`-th target q-function provided in :code:`q_targ_list`.

    Similarly, for non-discrete actions, the target is constructed as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,\min_{i,j}q_i(S_{t+n}, a_j(S_{t+n}))

    where :math:`a_i(s)` is the **mode** of the :math:`i`-th target policy provided in
    :code:`pi_targ_list`.


    where

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

    pi_targ_list : list of Policy, optional

        The list of policies that are used for constructing the TD-target. This is ignored if the
        action space is discrete and *required* otherwise.

    q_targ_list : list of Q

        The list of q-functions that are used for constructing the TD-target.

    optimizer : optax optimizer, optional

        An optax-style optimizer. The default optimizer is :func:`optax.adam(1e-3)
        <optax.adam>`.

    loss_function : callable, optional

        The loss function that will be used to regress to the (bootstrapped) target. The loss
        function is expected to be of the form:

        .. math::

            L(y_\text{true}, y_\text{pred}, w)\in\mathbb{R}

        where :math:`w>0` are sample weights. If left unspecified, this defaults to
        :func:`coax.value_losses.huber`. Check out the :mod:`coax.value_losses` module for other
        predefined loss functions.

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
            self, q, pi_targ_list=None, q_targ_list=None,
            optimizer=None, loss_function=None, policy_regularizer=None):

        if is_stochastic(q):
            raise NotImplementedError(f"{type(self).__name__} is not yet implement for StochasticQ")

        super().__init__(
            f=q,
            f_targ=None,
            optimizer=optimizer,
            loss_function=loss_function,
            policy_regularizer=policy_regularizer)

        self._check_input_lists(pi_targ_list, q_targ_list)
        del self._f_targ  # no need for this (only potential source of confusion)
        self.q_targ_list = q_targ_list
        self.pi_targ_list = [] if pi_targ_list is None else pi_targ_list

        # consistency check
        if isinstance(self.q.action_space, Discrete):
            if len(self.q_targ_list) < 2:
                raise ValueError("len(q_targ_list) must be at least 2")
        elif len(self.q_targ_list) * len(self.pi_targ_list) < 2:
            raise ValueError("len(q_targ_list) * len(pi_targ_list) must be at least 2")

        def loss_func(params, target_params, state, target_state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S = self.q.observation_preprocessor(next(rngs), transition_batch.S)
            A = self.q.action_preprocessor(next(rngs), transition_batch.A)
            W = jnp.clip(transition_batch.W, 0.1, 10.)  # clip importance weights to reduce variance

            # regularization term
            if self.policy_regularizer is None:
                regularizer = 0.
            else:
                # flip sign (typical example: regularizer = -beta * entropy)
                regularizer = -self.policy_regularizer.batch_eval(
                    target_params['reg'], target_params['reg_hparams'], target_state['reg'],
                    next(rngs), transition_batch)

            Q, state_new = self.q.function_type1(params, state, next(rngs), S, A, True)
            G = self.target_func(target_params, target_state, next(rngs), transition_batch)
            G += regularizer
            loss = self.loss_function(G, Q, W)

            dLoss_dQ = jax.grad(self.loss_function, argnums=1)
            td_error = -Q.shape[0] * dLoss_dQ(G, Q)  # e.g. (G - Q) if loss function is MSE

            # target-network estimate (is this worth computing?)
            Q_targ_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))
            for q, pm, st in qs:
                Q_targ, _ = q.function_type1(pm, st, next(rngs), S, A, False)
                assert Q_targ.ndim == 1, f"bad shape: {Q_targ.shape}"
                Q_targ_list.append(Q_targ)
            Q_targ_list = jnp.stack(Q_targ_list, axis=-1)
            assert Q_targ_list.ndim == 2, f"bad shape: {Q_targ_list.shape}"
            Q_targ = jnp.min(Q_targ_list, axis=-1)

            chex.assert_equal_shape([td_error, W, Q_targ])
            metrics = {
                f'{self.__class__.__name__}/loss': loss,
                f'{self.__class__.__name__}/td_error': jnp.mean(W * td_error),
                f'{self.__class__.__name__}/td_error_targ': jnp.mean(-dLoss_dQ(Q, Q_targ, W)),
            }
            return loss, (td_error, state_new, metrics)

        def grads_and_metrics_func(
                params, target_params, state, target_state, rng, transition_batch):

            rngs = hk.PRNGSequence(rng)
            grads, (td_error, state_new, metrics) = jax.grad(loss_func, has_aux=True)(
                params, target_params, state, target_state, next(rngs), transition_batch)

            # add some diagnostics about the gradients
            metrics.update(get_grads_diagnostics(grads, f'{self.__class__.__name__}/grads_'))

            return grads, state_new, metrics, td_error

        def td_error_func(params, target_params, state, target_state, rng, transition_batch):
            loss, (td_error, state_new, metrics) =\
                loss_func(params, target_params, state, target_state, rng, transition_batch)
            return td_error

        self._grads_and_metrics_func = jit(grads_and_metrics_func)
        self._td_error_func = jit(td_error_func)

    @property
    def q(self):
        return self._f

    @property
    def target_params(self):
        return hk.data_structures.to_immutable_dict({
            'q': self.q.params,
            'q_targ': [q.params for q in self.q_targ_list],
            'pi_targ': [pi.params for pi in self.pi_targ_list]})

    @property
    def target_function_state(self):
        return hk.data_structures.to_immutable_dict({
            'q': self.q.function_state,
            'q_targ': [q.function_state for q in self.q_targ_list],
            'pi_targ': [pi.function_state for pi in self.pi_targ_list]})

    def target_func(self, target_params, target_state, rng, transition_batch):
        rngs = hk.PRNGSequence(rng)

        # collect list of q-values
        if isinstance(self.q.action_space, Discrete):
            Q_sa_next_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))

            # compute A_next from q_i
            for q_i, params_i, state_i in qs:
                S_next = q_i.observation_preprocessor(next(rngs), transition_batch.S_next)
                Q_s_next, _ = q_i.function_type2(params_i, state_i, next(rngs), S_next, False)
                assert Q_s_next.ndim == 2, f"bad shape: {Q_s_next.shape}"
                A_next = (Q_s_next == Q_s_next.max(axis=1, keepdims=True)).astype(Q_s_next.dtype)
                A_next /= A_next.sum(axis=1, keepdims=True)  # there may be ties

                # evaluate on q_j
                for q_j, params_j, state_j in qs:
                    S_next = q_j.observation_preprocessor(next(rngs), transition_batch.S_next)
                    Q_sa_next, _ = q_j.function_type1(
                        params_j, state_j, next(rngs), S_next, A_next, False)
                    assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
                    f_inv = q_j.value_transform.inverse_func
                    Q_sa_next_list.append(f_inv(Q_sa_next))

        else:
            Q_sa_next_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))
            pis = list(zip(self.pi_targ_list, target_params['pi_targ'], target_state['pi_targ']))

            # compute A_next from pi_i
            for pi_i, params_i, state_i in pis:
                S_next = pi_i.observation_preprocessor(next(rngs), transition_batch.S_next)
                dist_params, _ = pi_i.function(params_i, state_i, next(rngs), S_next, False)
                A_next = pi_i.proba_dist.mode(dist_params)  # greedy action

                # evaluate on q_j
                for q_j, params_j, state_j in qs:
                    S_next = q_j.observation_preprocessor(next(rngs), transition_batch.S_next)
                    Q_sa_next, _ = q_j.function_type1(
                        params_j, state_j, next(rngs), S_next, A_next, False)
                    assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
                    f_inv = q_j.value_transform.inverse_func
                    Q_sa_next_list.append(f_inv(Q_sa_next))

        # take the min to mitigate over-estimation
        Q_sa_next_list = jnp.stack(Q_sa_next_list, axis=-1)
        assert Q_sa_next_list.ndim == 2, f"bad shape: {Q_sa_next_list.shape}"
        Q_sa_next = jnp.min(Q_sa_next_list, axis=-1)

        assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
        f = self.q.value_transform.transform_func
        return f(transition_batch.Rn + transition_batch.In * Q_sa_next)

    def _check_input_lists(self, pi_targ_list, q_targ_list):
        # check input: pi_targ_list
        if isinstance(self.q.action_space, Discrete):
            if pi_targ_list is not None:
                warnings.warn("pi_targ_list is ignored, because action space is discrete")
        else:
            if pi_targ_list is None:
                raise TypeError("pi_targ_list must be provided if action space is not discrete")
            if not isinstance(pi_targ_list, (tuple, list)):
                raise TypeError(
                    f"pi_targ_list must be a list or a tuple, got: {type(pi_targ_list)}")
            if len(pi_targ_list) < 1:
                raise ValueError("pi_targ_list cannot be empty")
            for pi_targ in pi_targ_list:
                if not is_policy(pi_targ):
                    raise TypeError(
                        f"all pi_targ in pi_targ_list must be a policies, got: {type(pi_targ)}")

        # check input: q_targ_list
        if not isinstance(q_targ_list, (tuple, list)):
            raise TypeError(f"q_targ_list must be a list or a tuple, got: {type(q_targ_list)}")
        if not q_targ_list:
            raise ValueError("q_targ_list cannot be empty")
        for q_targ in q_targ_list:
            if not isinstance(q_targ, Q):
                raise TypeError(f"all q_targ in q_targ_list must be a coax.Q, got: {type(q_targ)}")
