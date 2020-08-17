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

from ..utils import get_grads_diagnostics
from ._base import PolicyObjective


class VanillaPG(PolicyObjective):
    r"""
    A vanilla policy-gradient objective, a.k.a. REINFORCE-style objective.

    .. math::

        J(\theta; s,a)\ =\ \mathcal{A}(s,a)\,\log\pi_\theta(a|s)

    This objective has the property that its gradient with respect to
    :math:`\theta` yields the REINFORCE-style policy gradient.

    Parameters
    ----------
    pi : Policy

        The parametrized policy :math:`\pi_\theta(a|s)`.

    optimizer : optix optimizer, optional

        An optix-style optimizer. The default optimizer is :func:`optix.adam(1e-3)
        <jax.experimental.optix.adam>`.

    regularizer : PolicyRegularizer, optional

        A policy regularizer, see :mod:`coax.policy_regularizers`.

    """
    REQUIRES_PROPENSITIES = False

    def __init__(self, pi, optimizer=None, regularizer=None):
        super().__init__(pi=pi, optimizer=optimizer, regularizer=regularizer)
        self._init_funcs()

    def _init_funcs(self):

        def objective_func(params, state, rng, transition_batch, Adv):
            rngs = hk.PRNGSequence(rng)

            # get distribution params from function approximator
            S, A = transition_batch[:2]
            dist_params, state_new = self.pi.function(params, state, next(rngs), S, True)

            # compute REINFORCE-style objective
            A_raw = self.pi.proba_dist.preprocess_variate(A)
            log_pi = self.pi.proba_dist.log_proba(dist_params, A_raw)
            objective = Adv * log_pi

            # some consistency checks
            assert Adv.ndim == 1
            assert objective.ndim == 1

            # also pass auxiliary data to avoid multiple forward passes
            return objective, (dist_params, log_pi, state_new)

        def loss_func(params, state, rng, transition_batch, Adv, **reg_hparams):
            objective, (dist_params, log_pi, state_new) = \
                objective_func(params, state, rng, transition_batch, Adv)

            # flip sign to turn objective into loss
            loss = loss_bare = -jnp.mean(objective)

            # add regularization term
            if self.regularizer is not None:
                loss = loss + jnp.mean(self.regularizer.function(dist_params, **reg_hparams))

            # also pass auxiliary data to avoid multiple forward passes
            return loss, (loss, loss_bare, dist_params, log_pi, state_new)

        def grads_and_metrics_func(params, state, rng, transition_batch, Adv, **reg_hparams):
            grads, (loss, loss_bare, dist_params, log_pi, state_new) = \
                jax.grad(loss_func, has_aux=True)(
                    params, state, rng, transition_batch, Adv, **reg_hparams)

            name = self.__class__.__name__
            metrics = {f'{name}/loss': loss, f'{name}/loss_bare': loss_bare}

            # add sampled KL-divergence of the current policy relative to the behavior policy
            logP = transition_batch.logP  # log-propensities recorded from behavior policy
            metrics[f'{name}/kl_div_old'] = jnp.mean(jnp.exp(logP) * (logP - log_pi))

            # add some diagnostics of the gradients
            metrics.update(get_grads_diagnostics(grads, key_prefix=f'{name}/grads_'))

            # add regularization metrics
            if self.regularizer is not None:
                metrics.update(self.regularizer.metrics_func(dist_params, **reg_hparams))

            return grads, state_new, metrics

        self._grad_and_metrics_func = jax.jit(grads_and_metrics_func)
