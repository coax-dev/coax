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
from .._base.mixins import PolicyMixin
from ..utils import docstring, single_to_batch, batch_to_single
from ..proba_dists import CategoricalDist


__all__ = (
    'EpsilonGreedy',
    'BoltzmannPolicy',
)


class EpsilonGreedy(BaseFunc, PolicyMixin):
    r"""

    Derive an :math:`\epsilon`-greedy policy from a q-function.

    This policy samples actions :math:`a\sim\pi(.|s)` according to the following rule:

    .. math::

        u &\sim \text{Uniform([0, 1])} \\
        a_\text{rand} &\sim \text{Uniform}(\text{actions}) \\
        a\ &=\ \left\{\begin{matrix}
            a_\text{rand} & \text{ if } u < \epsilon \\
            \arg\max_{a'} q(s,a') & \text{ otherwise }
        \end{matrix}\right.

    Parameters
    ----------
    q : Q

        A state-action value function.

    epsilon : float between 0 and 1, optional

        The probability of sampling an action uniformly at random (as opposed to sampling greedily).

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, q, epsilon=0.1, random_seed=None):
        super().__init__(q.func_approx)
        self.q = q
        self.epsilon = epsilon
        self.random_seed = random_seed

        if not self.action_space_is_discrete:
            raise NotImplementedError(
                "EpsilonGreedy is not yet implemented for non-discrete action action spaces")

        self.proba_dist = CategoricalDist()
        self._init_funcs()

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        if hasattr(self.q.env, 'record_metrics'):
            self.q.env.record_metrics({f'{self.__class__.__name__}/epsilon': float(new_epsilon)})
        self._epsilon = new_epsilon

    @property
    def params(self):
        return self.q.params

    @property
    def function_state(self):
        return self.q.function_state

    @property
    def hyperparams(self):
        return {'epsilon': self.epsilon}

    @docstring(PolicyMixin.__call__)
    def __call__(self, s, return_logp=False):
        s = self.q.func_approx._preprocess_state(s)
        a, logp = self._sample_single_func(
            self.q.params, self.q.function_state, self.rng, s, self.epsilon)
        a = self.q.func_approx._postprocess_action(a)
        return (a, logp) if return_logp else a

    def greedy(self, s):
        s = self.q.func_approx._preprocess_state(s)
        a = self._mode_single_func(self.q.params, self.q.function_state, self.rng, s)
        return self.q.func_approx._postprocess_action(a)

    def dist_params(self, s):
        s = self.q.func_approx._preprocess_state(s)
        return self._apply_single_func(
            self.q.params, self.q.function_state, self.rng, s, self.epsilon)

    def batch_eval(self, S):
        dist_params, _ = self._apply_func(
            self.q.params, self.q.function_state, self.rng, S, False, self.epsilon)
        return dist_params

    def _init_funcs(self):

        def apply_func(params, state, rng, S, is_training, epsilon):
            Q_s, state_new = self.q.apply_func_type2(params, state, rng, S, is_training)
            P_greedy = Q_s == jnp.max(Q_s, axis=-1, keepdims=True)
            P_greedy /= P_greedy.sum(axis=-1, keepdims=True)
            P = epsilon / self.num_actions + (1 - epsilon) * P_greedy
            return {'logits': jnp.log(P + 1e-15)}, state_new

        def sample_func(params, state, rng, S, epsilon):
            rngs = hk.PRNGSequence(rng)
            dist_params, _ = apply_func(params, state, next(rngs), S, False, epsilon)
            X_a = self.proba_dist.sample(dist_params, next(rngs))
            logP = self.proba_dist.log_proba(dist_params, X_a)
            return X_a, logP

        def mode_func(params, state, rng, S):
            dist_params, _ = apply_func(params, state, rng, S, False, 0.)
            X_a = self.proba_dist.mode(dist_params)
            return X_a

        def apply_single_func(params, state, rng, s, epsilon):
            S = single_to_batch(s)
            dist_params, _ = apply_func(params, state, rng, S, False, epsilon)
            return batch_to_single(dist_params)

        def sample_single_func(params, state, rng, s, epsilon):
            rngs = hk.PRNGSequence(rng)
            S = single_to_batch(s)
            X_a, logP = sample_func(params, state, next(rngs), S, epsilon)
            A = self.action_postprocessor_func(params, next(rngs), X_a)
            a = batch_to_single(A)
            logp = batch_to_single(logP)
            return a, logp

        def mode_single_func(params, state, rng, s):
            rngs = hk.PRNGSequence(rng)
            S = single_to_batch(s)
            X_a = mode_func(params, state, next(rngs), S)
            A = self.action_postprocessor_func(params, next(rngs), X_a)
            a = batch_to_single(A)
            return a

        self._apply_func = jax.jit(apply_func, static_argnums=4)
        self._apply_single_func = jax.jit(apply_single_func)
        self._sample_func = jax.jit(sample_func)
        self._sample_single_func = jax.jit(sample_single_func)
        self._mode_func = jax.jit(mode_func)
        self._mode_single_func = jax.jit(mode_single_func)

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

        epsilon : float

            The epsilon that determines the probability of sampling uniformly at random.

        Returns
        -------
        dist_params : pytree with ndarray leaves

            A batch of conditional distribution parameters :math:`\pi(.|s)`. For instance, for a
            categorical distribution this would be ``{'logits': array([...])}``. For a normal
            distribution it is ``{'mu': array([...]), 'logvar': array([...])}``.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        """
        return self._apply_func

    @property
    def sample_func(self):
        r"""

        JIT-compiled function responsible for sampling a single action :math:`a` along with its
        corresponding log-propensity :math:`\log\pi(a|s)`. This function is used by the
        :attr:`sample` method.

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

        epsilon : float

            The epsilon that determines the probability of sampling uniformly at random.

        Returns
        -------
        X_a : transformed actions

            A batch of actions that are transformed in such a way that can be fed into
            :attr:`log_proba` method of the underlying :attr:`proba_dist`.

            Note that these actions cannot be fed directly into a gym-style environment. For
            example, if the action space is discrete, these transformed actions are (approximately)
            one-hot encoded. This means that we need to apply an :func:`argmax <coax.utils.argmax>`
            before we can feed the actions into a gym-style environment.

        logP : ndarray

            A batch of log-propensity associated with the sampled actions.

        """
        return self._sample_func

    @property
    def mode_func(self):
        r"""

        JIT-compiled function responsible for providing the mode of the conditional probability
        distribution :math:`\pi(.|s)` associated with a single state observation :math:`s`. This
        function is used by the :attr:`greedy` method.

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

        Returns
        -------
        X_a : transformed actions

            A batch of actions that are transformed in such a way that can be fed into
            :attr:`log_proba` method of the underlying :attr:`proba_dist`.

            Note that these actions cannot be fed directly into a gym-style environment. For
            example, if the action space is discrete, these transformed actions are (approximately)
            one-hot encoded. This means that we need to apply an :func:`argmax <coax.utils.argmax>`
            before we can feed the actions into a gym-style environment.

        """
        return self._mode_func


class BoltzmannPolicy(BaseFunc, PolicyMixin):
    r"""

    Derive a Boltzmann policy from a q-function.

    This policy samples actions :math:`a\sim\pi(.|s)` according to the following rule:

    .. math::

        p &= \text{softmax}(q(s,.) / \tau) \\
        a &\sim \text{Cat}(p)

    Note that this policy is only well-defined for *discrete* action spaces.

    Parameters
    ----------
    q : Q

        A state-action value function.

    tau : positive float, optional

        The so-called Boltzmann temperature :math:`\tau>0` sets the sharpness of the categorical
        distribution. Picking a small value for :math:`\tau` results in greedy sampling while large
        values results in uniform sampling.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, q, tau=1.0, random_seed=None):
        super().__init__(q.func_approx)
        self.q = q
        self.tau = tau
        self.random_seed = random_seed

        if not self.action_space_is_discrete:
            raise NotImplementedError(
                "BoltzmannPolicy is only well defined for non-discrete action "
                "spaces")

        self.proba_dist = CategoricalDist()
        self._init_funcs()

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, new_tau):
        if hasattr(self.q.env, 'record_metrics'):
            self.q.env.record_metrics(
                {f'{self.__class__.__name__}/tau': float(new_tau)})
        self._tau = new_tau

    @property
    def params(self):
        return self.q.params

    @property
    def function_state(self):
        return self.q.function_state

    @property
    def hyperparams(self):
        return {'tau': self.tau}

    @docstring(PolicyMixin.__call__)
    def __call__(self, s, return_logp=False):
        s = self.q.func_approx._preprocess_state(s)
        a, logp = self._sample_single_func(
            self.q.params, self.q.function_state, self.rng, s, self.tau)
        a = self.q.func_approx._postprocess_action(a)
        return (a, logp) if return_logp else a

    def greedy(self, s):
        s = self.q.func_approx._preprocess_state(s)
        a = self._mode_single_func(
            self.q.params, self.q.function_state, self.rng, s, self.tau)
        return self.q.func_approx._postprocess_action(a)

    def dist_params(self, s):
        s = self.q.func_approx._preprocess_state(s)
        return self._apply_single_func(
            self.q.params, self.q.function_state, self.rng, s, self.tau)

    def batch_eval(self, S):
        dist_params, _ = self._apply_func(
            self.q.params, self.q.function_state, self.rng, S, False, self.tau)
        return dist_params

    def _init_funcs(self):

        def apply_func(params, state, rng, S, is_training, tau):
            Q_s, state_new = self.q.apply_func_type2(params, state, rng, S, is_training)
            return {'logits': Q_s / tau}, state_new

        def sample_func(params, state, rng, S, tau):
            rngs = hk.PRNGSequence(rng)
            dist_params, _ = apply_func(params, state, next(rngs), S, False, tau)
            X_a = self.proba_dist.sample(dist_params, next(rngs))
            logP = self.proba_dist.log_proba(dist_params, X_a)
            return X_a, logP

        def mode_func(params, state, rng, S, tau):
            dist_params, _ = apply_func(params, state, rng, S, False, tau)
            X_a = self.proba_dist.mode(dist_params)
            return X_a

        def apply_single_func(params, state, rng, s, tau):
            S = single_to_batch(s)
            dist_params, _ = apply_func(params, state, rng, S, False, tau)
            return batch_to_single(dist_params)

        def sample_single_func(params, state, rng, s, tau):
            rngs = hk.PRNGSequence(rng)
            S = single_to_batch(s)
            X_a, logP = sample_func(params, state, next(rngs), S, tau)
            A = self.action_postprocessor_func(params, next(rngs), X_a)
            a = batch_to_single(A)
            logp = batch_to_single(logP)
            return a, logp

        def mode_single_func(params, state, rng, s, tau):
            rngs = hk.PRNGSequence(rng)
            S = single_to_batch(s)
            X_a = mode_func(params, state, next(rngs), S, tau)
            A = self.action_postprocessor_func(params, next(rngs), X_a)
            a = batch_to_single(A)
            return a

        self._apply_func = jax.jit(apply_func, static_argnums=4)
        self._apply_single_func = jax.jit(apply_single_func)
        self._sample_func = jax.jit(sample_func)
        self._sample_single_func = jax.jit(sample_single_func)
        self._mode_func = jax.jit(mode_func)
        self._mode_single_func = jax.jit(mode_single_func)

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

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        S : state observations

            A batch of state observations.

        tau : positive float

            The Boltzmann temperature :math:`\tau>0`.

        is_training : bool

            A flag that indicates whether we are in training mode.

        Returns
        -------
        dist_params : pytree with ndarray leaves

            A batch of conditional distribution parameters :math:`\pi(.|s)`. For instance, for a
            categorical distribution this would be ``{'logits': array([...])}``. For a normal
            distribution it is ``{'mu': array([...]), 'logvar': array([...])}``.

        state : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        """
        return self._apply_func

    @property
    def sample_func(self):
        r"""

        JIT-compiled function responsible for sampling a single action :math:`a` along with its
        corresponding log-propensity :math:`\log\pi(a|s)`. This function is used by the
        :attr:`sample` method.

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

        tau : positive float

            The Boltzmann temperature :math:`\tau>0`.

        Returns
        -------
        X_a : transformed actions

            A batch of actions that are transformed in such a way that can be fed into
            :attr:`log_proba` method of the underlying :attr:`proba_dist`.

            Note that these actions cannot be fed directly into a gym-style environment. For
            example, if the action space is discrete, these transformed actions are (approximately)
            one-hot encoded. This means that we need to apply an :func:`argmax <coax.utils.argmax>`
            before we can feed the actions into a gym-style environment.

        logP : ndarray

            A batch of log-propensity associated with the sampled actions.

        """
        return self._sample_func

    @property
    def mode_func(self):
        r"""

        JIT-compiled function responsible for providing the mode of the conditional probability
        distribution :math:`\pi(.|s)` associated with a single state observation :math:`s`. This
        function is used by the :attr:`greedy` method.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        state : pytree

            The internal state of the forward-pass function. See :attr:`Q.function_state
            <coax.Q.function_state>` and :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        S : state observations

            A batch of state observations.

        tau : positive float

            The Boltzmann temperature :math:`\tau>0`.

        Returns
        -------
        X_a : transformed actions

            A batch of actions that are transformed in such a way that can be fed into
            :attr:`log_proba` method of the underlying :attr:`proba_dist`.

            Note that these actions cannot be fed directly into a gym-style environment. For
            example, if the action space is discrete, these transformed actions are (approximately)
            one-hot encoded. This means that we need to apply an :func:`argmax <coax.utils.argmax>`
            before we can feed the actions into a gym-style environment.

        """
        return self._mode_func
