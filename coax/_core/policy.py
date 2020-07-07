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
import haiku as hk

from .._base.bases import BaseFunc
from .._base.mixins import ParamMixin, PolicyMixin
from ..utils import docstring, single_to_batch, batch_to_single
from ..proba_dists import CategoricalDist, NormalDist


class Policy(BaseFunc, ParamMixin, PolicyMixin):
    r"""

    A parametrized (i.e. learnable) policy :math:`\pi_\theta(a|s)`.

    Parameters
    ----------
    func_approx : function approximator

        This must be an instance of :class:`FuncApprox <coax.FuncApprox>` or a
        subclass thereof.

    proba_dist : ProbaDist, optional

        A probability distribution that is used to interpret the output of
        :attr:`func_approx.head_pi <FuncApprox.head_pi>`. Check out the
        :mod:`coax.proba_dists` module for available options.

        The default proba_dist is specified as the :attr:`default_proba_dist`
        property. For instance, for a discrete action space, the default
        proba_dist is the :func:`CategoricalDist
        <coax.proba_dists.CategoricalDist>`.

    random_seed : int, optional

        Sets the random state to get reproducible results.


    """
    COMPONENTS = (
        'body',
        'head_pi',
        'action_preprocessor',
        'action_postprocessor',
    )

    def __init__(
            self,
            func_approx,
            proba_dist=None,
            random_seed=None):

        super().__init__(func_approx)
        self.proba_dist = proba_dist or self.default_proba_dist
        self.random_seed = random_seed
        self._init_funcs()

    @docstring(PolicyMixin.__call__)
    def __call__(self, s, return_logp=False):
        s = self.func_approx._preprocess_state(s)
        assert self.env.observation_space.contains(s)
        a, logp = self._sample_single_func(self.params, self.function_state, self.rng, s)
        a = self.func_approx._postprocess_action(a)
        return (a, logp) if return_logp else a

    def dist_params(self, s):
        s = self.func_approx._preprocess_state(s)
        return self._apply_single_func(self.params, self.function_state, self.rng, s)

    def greedy(self, s):
        s = self.func_approx._preprocess_state(s)
        assert self.env.observation_space.contains(s)
        a = self._mode_single_func(self.params, self.function_state, self.rng, s)
        return self.func_approx._postprocess_action(a)

    def batch_eval(self, S):
        dist_params, _ = self._apply_func(self.params, self.function_state, self.rng, S, False)
        return dist_params

    @property
    def default_proba_dist(self):
        r"""

        The default probability distribution over the given action space
        :attr:`env.action_space`.


        """
        if self.action_space_is_discrete:
            return CategoricalDist()

        if self.action_space_is_box:
            return NormalDist()

        raise NotImplementedError(
            "no default policy distribution for action space of type: "
            f"{self.env.action_space.__class__.__name__}; please provide your "
            "own distribution by specifying proba_dist=...")

    def _init_funcs(self):

        def apply_func(params, state, rng, S, is_training):
            rngs = hk.PRNGSequence(rng)
            body = self.func_approx.apply_funcs['body']
            head = self.func_approx.apply_funcs['head_pi']
            state_new = state.copy()  # shallow copy
            X_s, state_new['body'] = body(params['body'], state['body'], next(rngs), S, is_training)
            dist_params, state_new['head_pi'] = \
                head(params['head_pi'], state['head_pi'], next(rngs), X_s, is_training)
            return dist_params, state_new

        def sample_func(params, state, rng, S):
            rngs = hk.PRNGSequence(rng)
            dist_params, _ = apply_func(params, state, next(rngs), S, False)
            X_a = self.proba_dist.sample(dist_params, next(rngs))
            logP = self.proba_dist.log_proba(dist_params, X_a)
            return X_a, logP

        def mode_func(params, state, rng, S):
            dist_params, _ = apply_func(params, state, rng, S, False)
            X_a = self.proba_dist.mode(dist_params)
            return X_a

        def apply_single_func(params, state, rng, s):
            S = single_to_batch(s)
            dist_params, _ = apply_func(params, state, rng, S, False)
            return batch_to_single(dist_params)

        def sample_single_func(params, state, rng, s):
            rngs = hk.PRNGSequence(rng)
            S = single_to_batch(s)
            X_a, logP = sample_func(params, state, next(rngs), S)
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

        JIT-compiled function responsible for the forward-pass through the
        underlying function approximator. This function is used by the
        :attr:`batch_eval` and :attr:`__call__` methods.

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
