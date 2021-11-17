import warnings

import jax
import jax.numpy as jnp
import optax
import haiku as hk

from .._core.policy import Policy
from ..utils import get_grads_diagnostics, jit
from ..regularizers import Regularizer


class PolicyObjective:
    r"""

    Abstract base class for policy objectives. To see a concrete example, have a look at
    :class:`coax.policy_objectives.VanillaPG`.

    Parameters
    ----------
    pi : Policy

        The parametrized policy :math:`\pi_\theta(a|s)`.

    regularizer : Regularizer, optional

        A policy regularizer, see :mod:`coax.regularizers`.

    """
    REQUIRES_PROPENSITIES = None

    def __init__(self, pi, optimizer=None, regularizer=None):
        if not isinstance(pi, Policy):
            raise TypeError(f"pi must be a Policy, got: {type(pi)}")
        if not isinstance(regularizer, (Regularizer, type(None))):
            raise TypeError(f"regularizer must be a Regularizer, got: {type(regularizer)}")

        self._pi = pi
        self._regularizer = regularizer

        # optimizer
        self._optimizer = optax.adam(1e-3) if optimizer is None else optimizer
        self._optimizer_state = self.optimizer.init(self._pi.params)

        def loss_func(params, state, hyperparams, rng, transition_batch, Adv):
            objective, (dist_params, log_pi, state_new) = \
                self.objective_func(params, state, hyperparams, rng, transition_batch, Adv)

            # flip sign to turn objective into loss
            loss = -objective

            # keep track of performance metrics
            metrics = {
                f'{self.__class__.__name__}/loss': loss,
                f'{self.__class__.__name__}/loss_bare': loss,
                f'{self.__class__.__name__}/kl_div_old':
                    jnp.mean(jnp.exp(transition_batch.logP) * (transition_batch.logP - log_pi)),
            }

            # add regularization term
            if self.regularizer is not None:
                hparams = hyperparams['regularizer']
                W = jnp.clip(transition_batch.W, 0.1, 10.)  # clip imp. weights to reduce variance
                regularizer, regularizer_metrics = self.regularizer.batch_eval(params,
                                                                               hparams,
                                                                               state,
                                                                               rng,
                                                                               transition_batch)
                loss = loss + jnp.mean(W * regularizer)
                metrics[f'{self.__class__.__name__}/loss'] = loss
                metrics.update({f'{self.__class__.__name__}/{k}': v for k,
                               v in regularizer_metrics.items()})

            # also pass auxiliary data to avoid multiple forward passes
            return loss, (metrics, state_new)

        def grads_and_metrics_func(params, state, hyperparams, rng, transition_batch, Adv):
            grads_func = jax.grad(loss_func, has_aux=True)
            grads, (metrics, state_new) = \
                grads_func(params, state, hyperparams, rng, transition_batch, Adv)

            # add some diagnostics of the gradients
            metrics.update(get_grads_diagnostics(grads, f'{self.__class__.__name__}/grads_'))

            return grads, state_new, metrics

        def apply_grads_func(opt, opt_state, params, grads):
            updates, new_opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_opt_state, new_params

        self._grad_and_metrics_func = jit(grads_and_metrics_func)
        self._apply_grads_func = jit(apply_grads_func, static_argnums=0)

    @property
    def pi(self):
        return self._pi

    @property
    def regularizer(self):
        return self._regularizer

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        new_optimizer_state_structure = jax.tree_structure(new_optimizer.init(self._f.params))
        if new_optimizer_state_structure != jax.tree_structure(self.optimizer_state):
            raise AttributeError("cannot set optimizer attr: mismatch in optimizer_state structure")
        self._optimizer = new_optimizer

    @property
    def optimizer_state(self):
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, new_optimizer_state):
        self._optimizer_state = new_optimizer_state

    @property
    def hyperparams(self):
        return hk.data_structures.to_immutable_dict({
            'regularizer': getattr(self.regularizer, 'hyperparams', {})})

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
        self.apply_grads(grads, function_state)
        return metrics

    def apply_grads(self, grads, function_state):
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
        return self._grad_and_metrics_func(
            self._pi.params, self._pi.function_state, self.hyperparams, self._pi.rng,
            transition_batch, Adv)
