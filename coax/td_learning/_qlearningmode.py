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

from .._base.mixins import PolicyMixin
from ..utils import get_magnitude_quantiles
from ._base import BaseTD


class QLearningMode(BaseTD):
    r"""

    An alternative to :class:`coax.td_learning.QLearning` that also works for continuous action
    spaces. The :math:`n`-step bootstrapped target is constructed as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,q_\text{targ}\left(S_{t+n}, a_\theta(s)\right)

    Here, :math:`a_\theta(s)` is the **mode** of the underlying conditional probability distribution
    :math:`\pi_\theta(.|s)`. In other words, we evaluate the policy as though the next action
    :math:`A_{t+n}` would have been the greedy action. This is equivalent to the usual q-learning
    bootstrap target implemented for discrete actions by :class:`coax.td_learning.QLearning`.

    Also, as usual, the :math:`n`-step reward and indicator are defined as:

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

    pi_targ : Policy

        The policy that is used for constructing the TD-target.

    q_targ : Q, optional

        The q-function that is used for constructing the TD-target. If this is left unspecified, we
        set ``q_targ = q`` internally.

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
    def __init__(self, q, pi_targ, q_targ=None, loss_function=None, value_transform=None):

        super().__init__(
            q=q, q_targ=q_targ, loss_function=loss_function, value_transform=value_transform)

        if q.qtype == 2:
            raise TypeError("q must be a type-1 q-function, got type-2")
        if not isinstance(pi_targ, PolicyMixin):
            raise TypeError(f"pi_targ must be a Policy, got: {type(pi_targ)}")

        self.pi_targ = pi_targ
        self._init_funcs()

    def _init_funcs(self):

        def q_apply_func_type1(θ, state_q, rng, S, X_a, is_training):
            """ type-I apply_func, except skipping the action_preprocessor """
            rngs = hk.PRNGSequence(rng)
            body = self.q.func_approx.apply_funcs['body']
            comb = self.q.func_approx.apply_funcs['state_action_combiner']
            head = self.q.func_approx.apply_funcs['head_q1']
            state_q_new = state_q.copy()
            X_s, state_q_new['body'] = body(θ['body'], state_q['body'], next(rngs), S, is_training)
            X_sa, state_q_new['state_action_combiner'] = comb(
                θ['state_action_combiner'], state_q['state_action_combiner'], next(rngs),
                X_s, X_a, is_training)
            Q_sa, state_q_new['head_q1'] = head(
                θ['head_q1'], state_q['head_q1'], next(rngs), X_sa, is_training)
            return jnp.squeeze(Q_sa, axis=1), state_q_new

        def target(θ_targ, θ_pi, state_q, state_pi, rng, Rn, In, S_next):
            rngs = hk.PRNGSequence(rng)
            dist_params, _ = self.pi_targ.apply_func(
                θ_pi, state_pi, next(rngs), S_next, False, **self.pi_targ.hyperparams)
            X_a_next = self.pi_targ.proba_dist.mode(dist_params)
            Q_next, _ = q_apply_func_type1(θ_targ, state_q, next(rngs), S_next, X_a_next, False)
            assert Q_next.ndim == 1
            f, f_inv = self.value_transform
            return f(Rn + In * f_inv(Q_next))

        def loss_func(θ, θ_targ, θ_pi, state_q, state_pi, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S, A, _, Rn, In, S_next, _, _ = transition_batch
            G = target(θ_targ, θ_pi, state_q, state_pi, next(rngs), Rn, In, S_next)
            Q, state_q_new = self.q.apply_func_type1(θ, state_q, next(rngs), S, A, True)
            loss = self.loss_function(G, Q)
            return loss, (loss, G, Q, S, A, state_q_new)

        def grads_and_metrics_func(θ, θ_targ, θ_pi, state_q, state_pi, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            grads, (loss, G, Q, S, A, state_q_new) = \
                jax.grad(loss_func, has_aux=True)(
                    θ, θ_targ, θ_pi, state_q, state_pi, next(rngs), transition_batch)

            # target-network estimate
            Q_targ, _ = self.q_targ.apply_func_type1(θ_targ, state_q, next(rngs), S, A, False)

            # residuals: estimate - better_estimate
            err = Q - G
            err_targ = Q_targ - Q

            name = self.__class__.__name__
            metrics = {
                f'{name}/loss': loss,
                f'{name}/bias': jnp.mean(err),
                f'{name}/rmse': jnp.sqrt(jnp.mean(jnp.square(err))),
                f'{name}/bias_targ': jnp.mean(err_targ),
                f'{name}/rmse_targ': jnp.sqrt(jnp.mean(jnp.square(err_targ)))}

            # add some diagnostics of the gradients
            metrics.update(get_magnitude_quantiles(grads, key_prefix=f'{name}/grads_'))

            return grads, state_q_new, metrics

        def td_error_func(θ, θ_targ, θ_pi, state_q, state_pi, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S, A, _, Rn, In, S_next, _, _ = transition_batch
            G = target(θ_targ, θ_pi, state_q, state_pi, next(rngs), Rn, In, S_next)
            Q, _ = self.q.apply_func_type1(θ, state_q, next(rngs), S, A, False)
            return G - Q

        self._grads_and_metrics_func = jax.jit(grads_and_metrics_func)
        self._td_error_func = jax.jit(td_error_func)

    def grads_and_metrics(self, transition_batch):
        return self.grads_and_metrics_func(
            self.q.params, self.q_targ.params, self.pi_targ.params, self.q.function_state,
            self.pi_targ.function_state, self.q.rng, transition_batch)

    def td_error(self, transition_batch):
        return self.td_error_func(
            self.q.params, self.q_targ.params, self.pi_targ.params, self.q.function_state,
            self.pi_targ.function_state, self.q.rng, transition_batch)

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

        target_params_q : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function to construct the
            bootstrapped TD-target.

        target_params_pi : pytree with ndarray leaves

            The model parameters (weights) used by the underlying target policy to construct the
            bootstrapped TD-target.

        state_q : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        state_pi : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        transition_batch : TransitionBatch

            A batch of transitions.

        Returns
        -------
        grads : pytree with ndarray leaves

            A pytree with the same structure as the input ``params_pi``.

        state_q : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return self._grads_and_metrics_func

    @property
    def td_error_func(self):
        r"""

        JIT-compiled function responsible for computing the TD-error. This
        function is used by the :attr:`td_error` method.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        target_params_q : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function to
            construct the bootstrapped TD-target.

        target_params_pi : pytree with ndarray leaves

            The model parameters (weights) used by the underlying target policy
            to construct the bootstrapped TD-target.

        state_q : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        state_pi : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

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
