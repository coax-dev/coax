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

from .._core.value_v import V
from ..utils import get_magnitude_quantiles
from ..value_transforms import ValueTransform
from ..value_losses import huber


class ValueTD:
    r"""

    TD-learning for state value functions :math:`v(s)`. The :math:`n`-step bootstrapped target is
    constructed as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,v_\text{targ}(S_{t+n})

    where

    .. math::

        R^{(n)}_t\ &=\ \sum_{k=0}^{n-1}\gamma^kR_{t+k} \\
        I^{(n)}_t\ &=\ \left\{\begin{matrix}
            0           & \text{if $S_{t+n}$ is a terminal state} \\
            \gamma^n    & \text{otherwise}
        \end{matrix}\right.


    Parameters
    ----------
    v : V

        The main state value function to update.

    v_targ : V, optional

        The state value function that is used for constructing the TD-target. If this is left
        unspecified, we set ``v_targ = v`` internally.

    loss_function : callable, optional

        The loss function that will be used to regress to the (bootstrapped) target. The loss
        function is expected to be of the form:

        .. math::

            L(y_\text{true}, y_\text{pred})\in\mathbb{R}

        If left unspecified, this defaults to :func:`coax.value_losses.huber`. Check out the
        :mod:`coax.value_losses` module for other predefined loss functions.

    value_transform : ValueTransform or pair of funcs, optional

        If provided, the returns are transformed as follows:

        .. math::

            G^{(n)}_t\ \mapsto\ f\left(G^{(n)}_t\right)\ =\
                f\left(R^{(n)}_t + I^{(n)}_t\,f^{-1}\left(v(S_{t+n})\right)\right)

        where :math:`f` and :math:`f^{-1}` are given by ``value_transform.transform_func`` and
        ``value_transform.inverse_func``, respectively. See :mod:`coax.td_learning` for examples of
        value-transforms. Note that a ValueTransform is just a glorified pair of functions, i.e.
        passing ``value_transform=(func, inverse_func)`` works just as well.

    """
    def __init__(self, v, v_targ=None, loss_function=None, value_transform=None):

        if not isinstance(v, V):
            raise TypeError(f"v must be a coax.V, got: {type(v)}")
        if not isinstance(v_targ, (V, type(None))):
            raise TypeError(
                f"v_targ must be a coax.V or None, got: {type(v_targ)}")

        self.v = v
        self.v_targ = v_targ or v
        self.loss_function = loss_function or huber
        if value_transform is None:
            self.value_transform = ValueTransform(lambda x: x, lambda x: x)
        else:
            self.value_transform = value_transform
        self._init_funcs()

    def _init_funcs(self):

        def target(params, state, rng, Rn, In, S_next):
            f, f_inv = self.value_transform
            V_next, _ = self.v_targ.apply_func(params, state, rng, S_next, False)
            return f(Rn + In * f_inv(V_next))

        def loss_func(params, target_params, state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S, _, _, Rn, In, S_next, _, _ = transition_batch
            G = target(target_params, state, next(rngs), Rn, In, S_next)
            V, state_new = self.v.apply_func(params, state, next(rngs), S, True)
            loss = self.loss_function(G, V)
            return loss, (loss, G, V, S, state_new)

        def grads_and_metrics_func(params, target_params, state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            grads, (loss, G, V, S, state_new) = \
                jax.grad(loss_func, has_aux=True)(
                    params, target_params, state, next(rngs), transition_batch)

            # target-network estimate
            V_targ, _ = self.v_targ.apply_func(target_params, state, next(rngs), S, False)

            # residuals: estimate - better_estimate
            err = V - G
            err_targ = V_targ - V

            name = self.__class__.__name__
            metrics = {
                f'{name}/loss': loss,
                f'{name}/bias': jnp.mean(err),
                f'{name}/rmse': jnp.sqrt(jnp.mean(jnp.square(err))),
                f'{name}/bias_targ': jnp.mean(err_targ),
                f'{name}/rmse_targ': jnp.sqrt(jnp.mean(jnp.square(err_targ)))}

            # add some diagnostics of the gradients
            metrics.update(get_magnitude_quantiles(grads, key_prefix=f'{name}/grads_'))

            return grads, state_new, metrics

        def td_error_func(params, target_params, state, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S, _, _, Rn, In, S_next, _, _ = transition_batch
            G = target(target_params, state, next(rngs), Rn, In, S_next)
            V, _ = self.v.apply_func(params, state, next(rngs), S, False)
            return G - V

        self._grads_and_metrics_func = jax.jit(grads_and_metrics_func)
        self._td_error_func = jax.jit(td_error_func)

    @property
    def hyperparams(self):
        return {}

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
        grads, state, metrics = self.grads_and_metrics(transition_batch)
        self.update_from_grads(grads, state)
        return metrics

    def update_from_grads(self, grads, state):
        r"""

        Update the model parameters (weights) of the underlying function approximator given
        pre-computed gradients.

        This method is useful in situations in which computation of the gradients is deligated to a
        separate (remote) process.

        Parameters
        ----------
        grads : pytree with ndarray leaves

            A batch of gradients, generated by the :attr:`grads` method.

        state : pytree

            The internal state of the forward-pass function. See :attr:`V.function_state
            <coax.V.function_state>` and :func:`haiku.transform_with_state` for more details.

        """
        self.v.func_approx.update_params(**grads)
        self.v.func_approx.update_function_state(**state)

    def grads_and_metrics(self, transition_batch):
        r"""

        Compute the gradients associated with a batch of transitions with
        corresponding advantages.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        grads : pytree with ndarray leaves

            A batch of gradients.

        state : pytree

            The internal state of the forward-pass function. See :attr:`V.function_state
            <coax.V.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return self.grads_and_metrics_func(
            self.v.params, self.v_targ.params, self.v.function_state, self.v.rng, transition_batch)

    def td_error(self, transition_batch):
        r"""

        Compute the TD-errors associated with a batch of transitions.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        td_errors : ndarray, shape: [batch_size]

            A batch of TD-errors.

        """
        return self.td_error_func(
            self.v.params, self.v_targ.params, self.v.function_state, self.v.rng, transition_batch)

    @property
    def grads_and_metrics_func(self):
        r"""

        JIT-compiled function responsible for computing the gradients, along with the updated
        internal state of the forward-pass function and some performance metrics. This function is
        used by the :attr:`update` method.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        target_params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function. This is used to
            construct the bootstrapped TD-target.

        state : pytree

            The internal state of the forward-pass function. See :attr:`V.function_state
            <coax.V.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        grads : pytree with ndarray leaves

            A pytree with the same structure as the input ``params``.

        state : pytree

            The internal state of the forward-pass function. See :attr:`V.function_state
            <coax.V.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return self._grads_and_metrics_func

    @property
    def td_error_func(self):
        r"""

        JIT-compiled function responsible for computing the TD-error. This function is used by the
        :attr:`td_error` method.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying
            q-function.

        target_params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying
            q-function. This is used to construct the bootstrapped TD-target.

        state : pytree

            The internal state of the forward-pass function. See :attr:`V.function_state
            <coax.V.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        td_errors : ndarray, shape: [batch_size]

            A batch of TD-errors.

        """
        return self._td_error_func
