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
import jax.numpy as jnp
import haiku as hk

from .._core.value_q import Q
from ..utils import get_grads_diagnostics
from ._base import PolicyObjective


class DeterministicPG(PolicyObjective):
    r"""
    A deterministic policy-gradient objective, a.k.a. DDPG-style objective. See
    :doc:`spinup:algorithms/ddpg` and references therein for more details.

    .. math::

        J(\theta; s,a)\ =\ q_\text{targ}(s, a_\theta(s))

    Here :math:`a_\theta(s)` is the *mode* of the underlying conditional
    probability distribution :math:`\pi_\theta(.|s)`. See e.g. the :attr:`mode`
    method of :class:`coax.proba_dists.NormalDist`. In other words, we evaluate
    the policy according to the current estimate of its best-case performance.

    This objective has the property that it explicitly maximizes the q-value.

    The full policy loss is constructed as:

    .. math::

        \text{loss}(\theta; s,a)\ =\
            -J(\theta; s,a)
            - \beta_\text{ent}\,H[\pi_\theta]
            + \beta_\text{kl-div}\,
                KL[\pi_{\theta_\text{prior}}, \pi_\theta]

    N.B. in order to unclutter the notation we abbreviated
    :math:`\pi(.|s)` by :math:`\pi`.

    Parameters
    ----------
    pi : Policy

        The parametrized policy :math:`\pi_\theta(a|s)`.

    q_targ : Q

        The target state-action value function :math:`q_\text{targ}(s,a)`.

    optimizer : optix optimizer, optional

        An optix-style optimizer. The default optimizer is :func:`optix.adam(1e-3)
        <jax.experimental.optix.adam>`.

    regularizer : PolicyRegularizer, optional

        A policy regularizer, see :mod:`coax.policy_regularizers`.

    """
    REQUIRES_PROPENSITIES = False

    def __init__(self, pi, q_targ, optimizer=None, regularizer=None):
        if not (isinstance(q_targ, Q) and q_targ.qtype == 1):
            raise TypeError(f"q must be a type-1 q-function, got: {type(q_targ)}")
        self.q_targ = q_targ
        super().__init__(pi=pi, optimizer=optimizer, regularizer=regularizer)
        self._init_funcs()

    def _init_funcs(self):

        def objective_func(params_pi, params_q, state_pi, state_q, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S = transition_batch.S

            # get distribution params from function approximator
            dist_params, state_pi_new = self.pi.function(params_pi, state_pi, next(rngs), S, True)

            # compute objective: q(s, a_greedy)
            A = self.pi.proba_dist.mode(dist_params)  # greedy action
            objective, _ = self.q_targ.function_type1(params_q, state_q, next(rngs), S, A, True)

            # some consistency checks
            assert objective.ndim == 1

            return objective, (dist_params, state_pi_new)

        def loss_func(params_pi, params_q, state_pi, state_q, rng, transition_batch, **reg_hparams):
            objective, (dist_params, state_pi_new) = \
                objective_func(params_pi, params_q, state_pi, state_q, rng, transition_batch)

            # flip sign to turn objective into loss
            loss = loss_bare = -jnp.mean(objective)

            # add regularization term
            if self.regularizer is not None:
                loss = loss + jnp.mean(self.regularizer.function(dist_params, **reg_hparams))

            # also pass auxiliary data to avoid multiple forward passes
            return loss, (loss, loss_bare, dist_params, state_pi_new)

        def grads_and_metrics_func(
                params_pi, params_q, state_pi, state_q, rng, transition_batch, **reg_hparams):

            rngs = hk.PRNGSequence(rng)
            grads, (loss, loss_bare, dist_params, state_pi_new) = \
                jax.grad(loss_func, has_aux=True)(
                    params_pi, params_q, state_pi, state_q, next(rngs), transition_batch,
                    **reg_hparams)

            name = self.__class__.__name__
            metrics = {f'{name}/loss': loss, f'{name}/loss_bare': loss_bare}

            # add sampled KL-divergence of the current policy relative to the behavior policy
            A = self.pi.proba_dist.preprocess_variate(transition_batch.A)
            log_pi = self.pi.proba_dist.log_proba(dist_params, A)
            logP = transition_batch.logP  # log-propensities recorded from behavior policy
            metrics[f'{name}/kl_div_old'] = jnp.mean(jnp.exp(logP) * (logP - log_pi))

            # add some diagnostics of the gradients
            metrics.update(get_grads_diagnostics(grads, key_prefix=f'{name}/grads_'))

            # add regularization metrics
            if self.regularizer is not None:
                metrics.update(self.regularizer.metrics_func(dist_params, **reg_hparams))

            return grads, state_pi_new, metrics

        self._grads_and_metrics_func = jax.jit(grads_and_metrics_func)

    def grads_and_metrics(self, transition_batch, Adv=None):
        r"""

        Compute the gradients associated with a batch of transitions with corresponding advantages.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Adv : ndarray, ignored

            This input is ignored; it is included for consistency with other policy objectives.

        Returns
        -------
        grads : pytree with ndarray leaves

            A batch of gradients.

        state : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return self.grads_and_metrics_func(
            self.pi.params, self.q_targ.params, self.pi.function_state, self.q_targ.function_state,
            self.pi.rng, transition_batch, **self.hyperparams)

    def update(self, transition_batch, Adv=None):
        r"""

        Update the model parameters (weights) of the underlying function approximator.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Adv : ndarray, ignored

            This input is ignored; it is included for consistency with other policy objectives.

        Returns
        -------
        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return super().update(transition_batch, Adv)

    @property
    def grads_and_metrics_func(self):
        r"""

        JIT-compiled function responsible for computing the gradients that are used to update the
        model parameters (weights) of underlying function approximator. This function is used by the
        :attr:`update` method.

        Parameters
        ----------
        params_pi : pytree with ndarray leaves

            The model parameters (weights) used by the underlying policy.

        params_q : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        state_pi : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        state_q : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        transition_batch : TransitionBatch

            A batch of transitions.

        \*\*reg_hparams

            Hyperparameters specific to the policy regularizer.

        Returns
        -------
        grads : pytree with ndarray leaves

            A pytree with the same structure as the input ``params_pi``.

        state_pi : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return self._grads_and_metrics_func
