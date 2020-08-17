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
from jax.experimental import optix

from .._core.policy import Policy
from ..policy_regularizers import PolicyRegularizer


class PolicyObjective:
    r"""

    Abstract base class for policy objectives. To see a concrete example, have a look at
    :class:`coax.policy_objectives.VanillaPG`.

    Parameters
    ----------
    pi : Policy

        The parametrized policy :math:`\pi_\theta(a|s)`.

    regularizer : PolicyRegularizer, optional

        A policy regularizer, see :mod:`coax.policy_regularizers`.

    """
    REQUIRES_PROPENSITIES = None

    def __init__(self, pi, optimizer, regularizer):
        if not isinstance(pi, Policy):
            raise TypeError(f"pi must be a Policy, got: {type(pi)}")
        if not isinstance(regularizer, (PolicyRegularizer, type(None))):
            raise TypeError(f"regularizer must be a PolicyRegularizer, got: {type(regularizer)}")

        self._pi = pi
        self._regularizer = regularizer

        # optimizer
        self._optimizer = optix.adam(1e-3) if optimizer is None else optimizer
        self._optimizer_state = self.optimizer.init(self._pi.params)

        # construct jitted param update function
        def apply_grads_func(opt, opt_state, params, grads):
            updates, new_opt_state = opt.update(grads, opt_state)
            new_params = optix.apply_updates(params, updates)
            return new_opt_state, new_params

        self._apply_grads_func = jax.jit(apply_grads_func, static_argnums=0)
        self._init_funcs()  # implemented downstream

    @property
    def pi(self):
        return self._pi

    @property
    def regularizer(self):
        return self._regularizer

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer_state(self):
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, new_optimizer_state):
        self._optimizer_state = new_optimizer_state

    @property
    def hyperparams(self):
        return getattr(self.regularizer, 'hyperparams', {})

    def update(self, transition_batch, Adv):
        r"""

        Update the model parameters (weights) of the underlying function approximator.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Adv : ndarray

            A batch of advantages :math:`\mathcal{A}(s,a)=q(s,a)-v(s)`.

        Returns
        -------
        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        grads, function_state, metrics = self.grads_and_metrics(transition_batch, Adv)
        if any(jnp.any(jnp.isnan(g)) for g in jax.tree_leaves(grads)):
            raise RuntimeError(f"found nan's in grads: {grads}")
        self.update_from_grads(grads, function_state)
        return metrics

    def update_from_grads(self, grads, function_state):
        r"""

        Update the model parameters (weights) of the underlying function approximator given
        pre-computed gradients.

        This method is useful in situations in which computation of the gradients is deligated to a
        separate (remote) process.

        Parameters
        ----------
        grads : pytree with ndarray leaves

            A batch of gradients, generated by the :attr:`grads` method.

        function_state : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        """
        self._pi.function_state = function_state
        self.optimizer_state, self._pi.params = \
            self._apply_grads_func(self.optimizer, self.optimizer_state, self._pi.params, grads)

    def grads_and_metrics(self, transition_batch, Adv):
        r"""

        Compute the gradients associated with a batch of transitions with
        corresponding advantages.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Adv : ndarray

            A batch of advantages :math:`\mathcal{A}(s,a)=q(s,a)-v(s)`.

        Returns
        -------
        grads : pytree with ndarray leaves

            A batch of gradients.

        function_state : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        if self.REQUIRES_PROPENSITIES and jnp.all(transition_batch.logP == 0):
            warnings.warn(
                f"In order for {self.__class__.__name__} to work properly, transition_batch.logP "
                "should be non-zero. Please sample actions with their propensities: "
                "a, logp = pi(s, return_logp=True) and then add logp to your reward tracer, "
                "e.g. nstep_tracer.add(s, a, r, done, logp)")
        return self.grad_and_metrics_func(
            self._pi.params, self._pi.function_state, self._pi.rng, transition_batch, Adv,
            **self.hyperparams)

    @property
    def grad_and_metrics_func(self):
        r"""

        JIT-compiled function responsible for computing the gradients, along with the updated
        internal state of the forward-pass function and some performance metrics. This function is
        used by the :attr:`update` method.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying policy.

        function_state : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        transition_batch : TransitionBatch

            A batch of transitions. Note that ``transition_batch.logP`` cannot
            be ``None``, as it is required by the PPO-clip objective.

        Adv : ndarray

            A batch of advantages :math:`\mathcal{A}(s,a)=q(s,a)-v(s)`.

        \*\*hyperparams

            Hyperparameters specific to the objective, see :attr:`hyperparams`.

        Returns
        -------
        grads : pytree with ndarray leaves

            A batch of gradients.

        function_state : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.


        """
        return self._grad_and_metrics_func
