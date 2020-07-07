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

from .._base.bases import BaseFunc
from .._base.mixins import ParamMixin
from ..value_losses import huber
from ..utils import single_to_batch, batch_to_single
from .episodic_cache import NStepCache


__all__ = (
    'V',
)


class V(BaseFunc, ParamMixin):
    r"""

    A state value function :math:`v(s)`.

    Parameters
    ----------
    func_approx : function approximator

        This must be an instance of :class:`FuncApprox <coax.FuncApprox>` or a subclass thereof.

    gamma : float between 0 and 1, optional

        The future-discount factor :math:`\gamma\in[0,1]`.

    n : positive int, optional

        The number of time steps over which to do :math:`n`-step bootstrapping.

    bootstrap_with_params_copy : bool, optional

        Whether to use a separate copy of the model weights (known as a *target network*) to
        construct the bootstrapped TD-target.

        If this is set to ``True`` you must remember to periodically call the
        :func:`sync_params_copy` method.

    loss_function : function, optional

        The loss function that will be used to regress to the (bootstrapped) target. The loss
        function is expected to be of the form:

        .. math::

            L(y_\text{true}, y_\text{pred})\in\mathbb{R}

        Check out the :mod:`coax._core.losses` module for some predefined loss functions. If left
        unspecified, this defaults to the :func:`Huber <coax._core.value_losses.Huber>` loss
        function.

    """
    COMPONENTS = ('body', 'head_v')

    def __init__(
            self,
            func_approx,
            gamma=0.9,
            n=1,
            bootstrap_with_params_copy=False,
            loss_function=None):

        super().__init__(func_approx)
        self.gamma = float(gamma)
        self.n = int(n)
        self.bootstrap_with_params_copy = bool(bootstrap_with_params_copy)
        self.loss_function = loss_function or huber

        self._cache = NStepCache(self.env, self.n, self.gamma)
        self._init_funcs()

    def __call__(self, s, use_params_copy=False):
        r"""

        Evaluate the value function on a state observation :math:`s`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        use_params_copy : bool, optional

            Whether to use the target network weights, i.e. the model weights contained in
            :attr:`func_approx.params_copy <coax.FuncApprox.params_copy>` instead of
            :attr:`func_approx.params <coax.FuncApprox.params>`.

            If this is set to ``True`` you must remember to periodically call the
            :func:`sync_params_copy` method.

        Returns
        -------
        v : ndarray

            The output of the function-approximator's :attr:`func_approx.head_v
            <coax.FuncApprox.head_v>`.

        """
        s = self.func_approx._preprocess_state(s)
        v = self._apply_single_func(self.params, self.function_state, self.rng, s)
        return v

    def batch_eval(self, S, use_params_copy=False):
        r"""

        Evaluate the value function on a batch of state observations.

        Parameters
        ----------
        S : ndarray

            A batch of state observations :math:`s`.

        use_params_copy : bool, optional

            Whether to use the target network weights, i.e. the model weights contained in
            :attr:`func_approx.params_copy <coax.FuncApprox.params_copy>` instead of
            :attr:`func_approx.params <coax.FuncApprox.params>`.

            If this is set to ``True`` you must remember to periodically call the
            :func:`sync_params_copy` method.

        Returns
        -------
        V : ndarray

            The output of the function-approximator's :attr:`func_approx.head_v
            <coax.FuncApprox.head_v>`.

        """
        V, _ = self.apply_func(self.params, self.function_state, self.rng, S, False)
        return V

    def _init_funcs(self):

        def apply_func(params, state, rng, S, is_training):
            rngs = hk.PRNGSequence(rng)
            body = self.func_approx.apply_funcs['body']
            head = self.func_approx.apply_funcs['head_v']
            state_new = state.copy()  # shallow copy
            X_s, state_new['body'] = \
                body(params['body'], state['body'], next(rngs), S, is_training)
            V, state_new['head_v'] = \
                head(params['head_v'], state['head_v'], next(rngs), X_s, is_training)
            return jnp.squeeze(V, axis=1), state_new

        def apply_single_func(params, state, rng, s):
            S = single_to_batch(s)
            V, _ = apply_func(params, state, rng, S, is_training=False)
            v = batch_to_single(V)
            return v

        self._apply_func = jax.jit(apply_func, static_argnums=4)
        self._apply_single_func = jax.jit(apply_single_func)

    @property
    def apply_func(self):
        r"""

        JIT-compiled function responsible for the forward-pass through the underlying function
        approximator. This function is used by the :attr:`batch_eval` and :attr:`__call__` methods.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        S : state observations

            A batch of state observations.

        is_training : bool

            A flag that indicates whether we are in training mode.

        Returns
        -------
        V : ndarray

            A batch of state values :math:`v(s)`.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        """
        return self._apply_func
