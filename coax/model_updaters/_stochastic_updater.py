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
import optax

from .._core.dynamics_model import DynamicsModel
from .._core.reward_model import RewardModel
from ..utils import get_grads_diagnostics
from ..value_losses import huber
from ..regularizers import Regularizer


__all__ = (
    'StochasticUpdater',
)


class StochasticUpdater:
    r"""

    Model updater that uses *sampling* for maximum-likelihood estimation.

    Parameters
    ----------
    model : DynamicsModel or RewardModel

        The main dynamics/reward model to update.

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


    """
    def __init__(self, model, optimizer=None, loss_function=None, regularizer=None):
        if not isinstance(model, (DynamicsModel, RewardModel)):
            raise TypeError(f"model must be a DynamicsModel or RewardModel, got: {type(model)}")
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
            S = self.model.observation_preprocessor(transition_batch.S)
            A = self.model.action_preprocessor(transition_batch.A)
            dist_params, new_state = \
                self.model.function_type1(params, state, next(rngs), S, A, True)
            y_pred = self.model.proba_dist.sample(dist_params, next(rngs))

            if isinstance(self.model, DynamicsModel):
                y_true = self.model.observation_preprocessor(transition_batch.S_next)
            elif isinstance(self.model, RewardModel):
                y_true = self.model.value_transform.transform_func(transition_batch.Rn)
            else:
                raise AssertionError(f"unexpected model type: {type(self.model)}")

            loss = self.loss_function(y_true, y_pred)
            td_error = -jax.grad(self.loss_function, argnums=1)(y_true, y_pred)

            # add regularization term
            if self.regularizer is not None:
                hparams = hyperparams['regularizer']
                loss = loss + jnp.mean(self.regularizer.function(dist_params, **hparams))

            return loss, (loss, td_error, new_state)

        def grads_and_metrics_func(params, state, hyperparams, rng, transition_batch):
            grads, (loss, td_error, new_state) = \
                jax.grad(loss_func, has_aux=True)(params, state, hyperparams, rng, transition_batch)

            name = self.__class__.__name__
            metrics = {
                f'{name}/loss': loss,
                f'{name}/td_error': jnp.mean(td_error),
            }

            # add some diagnostics of the gradients
            metrics.update(get_grads_diagnostics(grads, key_prefix=f'{name}/grads_'))

            return grads, new_state, metrics

        self._apply_grads_func = jax.jit(apply_grads_func, static_argnums=0)
        self._grads_and_metrics_func = jax.jit(grads_and_metrics_func)

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