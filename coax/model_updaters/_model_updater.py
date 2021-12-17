import jax
import jax.numpy as jnp
import haiku as hk
import optax

from ..utils import (
    get_grads_diagnostics, is_stochastic, is_reward_function, is_transition_model, jit)
from ..value_losses import huber
from ..regularizers import Regularizer


__all__ = (
    'ModelUpdater',
)


class ModelUpdater:
    r"""

    Model updater that uses *sampling* for maximum-likelihood estimation.

    Parameters
    ----------
    model : [Stochastic]TransitionModel or [Stochastic]RewardFunction

        The main dynamics model to update.

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

    regularizer : Regularizer, optional

        A stochastic regularizer, see :mod:`coax.regularizers`.

    """
    def __init__(self, model, optimizer=None, loss_function=None, regularizer=None):
        if not (is_reward_function(model) or is_transition_model(model)):
            raise TypeError(f"model must be a dynamics model, got: {type(model)}")
        if not isinstance(regularizer, (Regularizer, type(None))):
            raise TypeError(f"regularizer must be a Regularizer, got: {type(regularizer)}")

        self.model = model
        self.loss_function = huber if loss_function is None else loss_function
        self.regularizer = regularizer

        # optimizer
        self._optimizer = optax.adam(1e-3) if optimizer is None else optimizer
        self._optimizer_state = self.optimizer.init(self.model.params)

        def apply_grads_func(opt, opt_state, params, grads):
            updates, new_opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_opt_state, new_params

        def loss_func(params, state, hyperparams, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S = self.model.observation_preprocessor(next(rngs), transition_batch.S)
            A = self.model.action_preprocessor(next(rngs), transition_batch.A)
            W = jnp.clip(transition_batch.W, 0.1, 10.)  # clip importance weights to reduce variance

            if is_stochastic(self.model):
                dist_params, new_state = \
                    self.model.function_type1(params, state, next(rngs), S, A, True)
                y_pred = self.model.proba_dist.sample(dist_params, next(rngs))
            else:
                y_pred, new_state = self.model.function_type1(params, state, next(rngs), S, A, True)

            if is_transition_model(self.model):
                y_true = self.model.observation_preprocessor(next(rngs), transition_batch.S_next)
            elif is_reward_function(self.model):
                y_true = self.model.value_transform.transform_func(transition_batch.Rn)
            else:
                raise AssertionError(f"unexpected model type: {type(self.model)}")

            loss = self.loss_function(y_true, y_pred, W)
            metrics = {
                f'{self.__class__.__name__}/loss': loss,
                f'{self.__class__.__name__}/loss_bare': loss,
            }

            # add regularization term
            if self.regularizer is not None:
                hparams = hyperparams['regularizer']
                loss = loss + jnp.mean(W * self.regularizer.function(dist_params, **hparams))
                metrics[f'{self.__class__.__name__}/loss'] = loss
                metrics.update(self.regularizer.metrics_func(dist_params, **hparams))

            return loss, (metrics, new_state)

        def grads_and_metrics_func(params, state, hyperparams, rng, transition_batch):
            grads, (metrics, new_state) = \
                jax.grad(loss_func, has_aux=True)(params, state, hyperparams, rng, transition_batch)

            # add some diagnostics of the gradients
            metrics.update(get_grads_diagnostics(grads, f'{self.__class__.__name__}/grads_'))

            return grads, new_state, metrics

        self._apply_grads_func = jit(apply_grads_func, static_argnums=0)
        self._grads_and_metrics_func = jit(grads_and_metrics_func)

    def update(self, transition_batch):
        r"""

        Update the model parameters (weights) of the underlying function approximator.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        grads, function_state, metrics = self.grads_and_metrics(transition_batch)
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

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        """
        self.model.function_state = function_state
        self.optimizer_state, self.model.params = \
            self._apply_grads_func(self.optimizer, self.optimizer_state, self.model.params, grads)

    def grads_and_metrics(self, transition_batch):
        r"""

        Compute the gradients associated with a batch of transitions.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        grads : pytree with ndarray leaves

            A batch of gradients.

        function_state : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return self._grads_and_metrics_func(
            self.model.params, self.model.function_state, self.hyperparams, self.model.rng,
            transition_batch)

    @property
    def hyperparams(self):
        return hk.data_structures.to_immutable_dict({
            'regularizer': getattr(self.regularizer, 'hyperparams', {})})

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        new_optimizer_state_structure = jax.tree_structure(new_optimizer.init(self.model.params))
        if new_optimizer_state_structure != jax.tree_structure(self.optimizer_state):
            raise AttributeError("cannot set optimizer attr: mismatch in optimizer_state structure")
        self._optimizer = new_optimizer

    @property
    def optimizer_state(self):
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, new_optimizer_state):
        if jax.tree_structure(new_optimizer_state) != jax.tree_structure(self.optimizer_state):
            raise AttributeError("cannot set optimizer_state attr: mismatch in tree structure")
        self._optimizer_state = new_optimizer_state
