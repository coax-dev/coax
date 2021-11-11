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

from functools import partial
import warnings

import jax
import jax.numpy as jnp
import haiku as hk
import chex
from gym.spaces import Discrete

from ..utils import is_policy, is_qfunction, jit, is_stochastic, single_to_batch, stack_trees
from ._doubleqlearning import DoubleQLearning


class ClippedDoubleQLearning(DoubleQLearning):
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

        self._check_input_lists(q, pi_targ_list, q_targ_list)
        self.q_targ_list = q_targ_list
        self.pi_targ_list = [] if pi_targ_list is None else pi_targ_list

        super().__init__(
            q=q,
            pi_targ=None if not pi_targ_list else pi_targ_list[0],
            q_targ=q_targ_list[0],
            optimizer=optimizer,
            loss_function=loss_function,
            policy_regularizer=policy_regularizer)

        # consistency check
        if isinstance(self.q.action_space, Discrete):
            if len(self.q_targ_list) < 2:
                raise ValueError("len(q_targ_list) must be at least 2")
        elif len(self.q_targ_list) * len(self.pi_targ_list) < 2:
            raise ValueError("len(q_targ_list) * len(pi_targ_list) must be at least 2")

        # persist DoubleQLearning jit functions
        self._super_grads_and_metrics_func = self._grads_and_metrics_func
        self._super_td_error_func = self._td_error_func

        def q_targ_pi_argmin(target_params, target_state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)

            def q_targ_target_func(p, s):
                return self.q_targ_func(p, s, rng=next(rngs), transition_batch=transition_batch)
            targets = jax.vmap(q_targ_target_func)(*stack_trees(target_params, target_state))
            if is_stochastic(self.q_targ):
                targets = jax.vmap(self.q_targ.proba_dist.mean)(targets)
                Q_sa_targets = jax.vmap(lambda t: self.q_targ.proba_dist.postprocess_variate(
                    next(rngs), t, batch_mode=True))(targets)
            else:
                Q_sa_targets = targets
            assert Q_sa_targets.ndim == 2
            return jnp.argmin(Q_sa_targets, axis=0)

        def grads_and_metrics_func(
                params, target_params, state, target_state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            # compute argmin target q-function per transition
            q_targ_argmin = q_targ_pi_argmin(
                target_params, target_state, next(rngs), transition_batch)

            def apply_params(q_targ_idx, transition, p, s):
                # extract target q-function parameters and state
                t_params = jax.tree_util.tree_map(lambda t: t[q_targ_idx], p)
                t_state = jax.tree_util.tree_map(lambda t: t[q_targ_idx], s)
                return self._super_grads_and_metrics_func(params, t_params,
                                                          state, t_state,
                                                          next(rngs), single_to_batch(transition))
            single_grads_and_metrics = jax.vmap(
                apply_params, in_axes=(0, 0, None, None), out_axes=0)
            # have to average because the grads are computed using a single transition
            batch_grads, batch_state_new, batch_metrics, batch_td_error = jax.tree_util.tree_map(
                lambda t: jnp.mean(t, axis=0), single_grads_and_metrics(
                    q_targ_argmin, transition_batch, *stack_trees(target_params, target_state))
            )

            return batch_grads, batch_state_new, batch_metrics, batch_td_error

        def td_error_func(params, target_params, state, target_state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            q_targ_argmin = q_targ_pi_argmin(
                target_params, target_state, next(rngs), transition_batch)

            def apply_params(q_targ_idx, transition, p, s):
                t_params = jax.tree_util.tree_map(lambda t: t[q_targ_idx], p)
                t_state = jax.tree_util.tree_map(lambda t: t[q_targ_idx], s)
                return self._super_td_error_func(params, t_params,
                                                 state, t_state,
                                                 next(rngs), single_to_batch(transition))
            single_td_error = jax.vmap(apply_params, in_axes=(0, 0, None, None), out_axes=0)
            batch_td_error = jax.tree_util.tree_map(
                lambda t: jnp.mean(t, axis=0),
                single_td_error(q_targ_argmin, transition_batch,
                                *stack_trees(target_params, target_state)))
            return batch_td_error

        self._grads_and_metrics_func = jit(grads_and_metrics_func)
        self._td_error_func = jit(td_error_func)

    @property
    def target_params(self):
        if self.pi_targ_list:
            return tuple([hk.data_structures.to_immutable_dict({
                'q': self.q.params,
                'q_targ': q_targ.params,
                'pi_targ': pi_targ.params
            }) for q_targ in self.q_targ_list for pi_targ in self.pi_targ_list])
        else:
            return tuple([hk.data_structures.to_immutable_dict({
                'q': self.q.params,
                'q_targ': q_targ.params
            }) for q_targ in self.q_targ_list])

    @property
    def target_function_state(self):
        if self.pi_targ_list:
            return tuple([hk.data_structures.to_immutable_dict({
                'q': self.q.function_state,
                'q_targ': q_targ.function_state,
                'pi_targ': pi_targ.function_state
            }) for q_targ in self.q_targ_list for pi_targ in self.pi_targ_list])
        else:
            return tuple([hk.data_structures.to_immutable_dict({
                'q': self.q.function_state,
                'q_targ': q_targ.function_state
            }) for q_targ in self.q_targ_list])

    def _check_input_lists(self, q, pi_targ_list, q_targ_list):
        # check input: pi_targ_list
        if isinstance(q.action_space, Discrete):
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
            if not is_qfunction(q_targ):
                raise TypeError(
                    f"all q_targ in q_targ_list must be q-functions, got: {type(q_targ)}")
        if len(q_targ_list) > 1:
            chex.assert_trees_all_equal_shapes(*[q_targ.params for q_targ in q_targ_list])
            chex.assert_trees_all_equal_shapes(*[q_targ.function_state for q_targ in q_targ_list])
