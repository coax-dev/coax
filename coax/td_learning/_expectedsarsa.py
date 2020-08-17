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

import gym
import jax
import jax.numpy as jnp
import haiku as hk

from .._core.base_policy import PolicyMixin
from ..utils import get_grads_diagnostics
from ._base import BaseTDLearningQ


class ExpectedSarsa(BaseTDLearningQ):
    r"""

    TD-learning with expected-SARSA updates. The :math:`n`-step bootstrapped target is constructed
    as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t
            + I^{(n)}_t\,\mathop{\mathbb{E}}_{a\sim\pi_\text{targ}(.|S_{t+n})}\,
                q_\text{targ}\left(S_{t+n}, a\right)

    Note that ordinary :class:`SARSA <coax.td_learning.Sarsa>` target is the sampled estimate of the
    above target.

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

    optimizer : optix optimizer, optional

        An optix-style optimizer. The default optimizer is :func:`optix.adam(1e-3)
        <jax.experimental.optix.adam>`.

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
    def __init__(
            self, q, pi_targ, q_targ=None, optimizer=None,
            loss_function=None, value_transform=None):

        if not isinstance(q.action_space, gym.spaces.Discrete):
            raise NotImplementedError(
                f"{self.__class__.__name__} class is only implemented for discrete actions spaces; "
                "you can use Sarsa or QLearningMode for non-discrete action spaces")
        if not isinstance(pi_targ, PolicyMixin):
            raise TypeError(f"pi_targ must be a Policy, got: {type(pi_targ)}")

        self.pi_targ = pi_targ
        super().__init__(
            q=q, q_targ=q_targ, optimizer=optimizer,
            loss_function=loss_function, value_transform=value_transform)

    def _init_funcs(self):

        def target(θ_targ, θ_pi, state_q, state_pi, rng, Rn, In, S_next):
            rngs = hk.PRNGSequence(rng)
            dist_params, _ = self.pi_targ.function(θ_pi, state_pi, next(rngs), S_next, False)
            P = jax.nn.softmax(dist_params['logits'], axis=-1)
            Q_s_next, _ = self.q_targ.function_type2(θ_targ, state_q, next(rngs), S_next, False)
            assert P.ndim == 2
            assert Q_s_next.ndim == 2
            Q_sa_next = jnp.einsum('ij,ij->i', P, Q_s_next)
            f, f_inv = self.value_transform
            return f(Rn + In * f_inv(Q_sa_next))

        def loss_func(θ, θ_targ, θ_pi, state_q, state_pi, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S, A, _, Rn, In, S_next, _, _ = transition_batch
            A = self.q.action_preprocessor(A)
            G = target(θ_targ, θ_pi, state_q, state_pi, next(rngs), Rn, In, S_next)
            Q, state_q_new = self.q.function_type1(θ, state_q, next(rngs), S, A, True)
            loss = self.loss_function(G, Q)
            return loss, (loss, G, Q, S, A, state_q_new)

        def grads_and_metrics_func(θ, θ_targ, θ_pi, state_q, state_pi, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            grads, (loss, G, Q, S, A, state_q_new) = \
                jax.grad(loss_func, has_aux=True)(
                    θ, θ_targ, θ_pi, state_q, state_pi, next(rngs), transition_batch)

            # target-network estimate
            Q_targ, _ = self.q_targ.function_type1(θ_targ, state_q, next(rngs), S, A, False)

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
            metrics.update(get_grads_diagnostics(grads, key_prefix=f'{name}/grads_'))

            return grads, state_q_new, metrics

        def td_error_func(θ, θ_targ, θ_pi, state_q, state_pi, rng, transition_batch):
            rngs = hk.PRNGSequence(rng)
            S, A, _, Rn, In, S_next, _, _ = transition_batch
            A = self.q.action_preprocessor(A)
            G = target(θ_targ, θ_pi, state_q, state_pi, next(rngs), Rn, In, S_next)
            Q, _ = self.q.function_type1(θ, state_q, next(rngs), S, A, False)
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
