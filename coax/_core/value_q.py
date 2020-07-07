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
from ..utils import single_to_batch, batch_to_single


__all__ = (
    'Q',
)


class Q(BaseFunc, ParamMixin):
    r"""

    A state-action value function :math:`q(s,a)`.

    Parameters
    ----------
    func_approx : function approximator

        This must be an instance of :class:`FuncApprox <coax.FuncApprox>` or a
        subclass thereof.

    qtype : 1 or 2, optional

        Whether to model the value function as a **type-I** q-function:

        .. math::

            (s, a)\ \mapsto\ q(s,a) \in \mathbb{R}

        or as a **type-II** q-function:

        .. math::

            s\ \mapsto\ q(s,.) \in \mathbb{R}^n

        Here :math:`n` is the number of discrete actions. Naturally, this only
        applies to discrete action spaces.

    """
    COMPONENTS_TYPE1 = (
        'body',
        'head_q1',
        'action_preprocessor',
        'action_postprocessor',
        'state_action_combiner',
    )
    COMPONENTS_TYPE2 = (
        'body',
        'head_q2',
        'action_preprocessor',
        'action_postprocessor',
    )

    def __init__(self, func_approx, qtype=1):
        super().__init__(func_approx)
        self._init_qtype(qtype)  # sets self._qtype and self.COMPONENTS
        self._init_funcs()

    @property
    def qtype(self):
        return self._qtype

    def __call__(self, s, a=None):
        r"""

        Evaluate the state-action function on a state observation :math:`s` or
        on a state-action pair :math:`(s, a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action

            A single action :math:`a`.

        Returns
        -------
        q_sa or q_s : ndarray

            Depending on whether ``a`` is provided, this either returns a scalar representing
            :math:`q(s,a)\in\mathbb{R}` or a vector representing :math:`q(s,.)\in\mathbb{R}^n`,
            where :math:`n` is the number of discrete actions. Naturally, this only applies for
            discrete action spaces.

        """
        s = self.func_approx._preprocess_state(s)
        assert self.env.observation_space.contains(s), f"bad state: {s}"
        if a is None:
            return self._apply_single_type2_func(self.params, self.function_state, self.rng, s)

        assert self.env.action_space.contains(a), f"bad action: {a}"
        return self._apply_single_type1_func(self.params, self.function_state, self.rng, s, a)

    def batch_eval(self, S, A=None):
        r"""

        Evaluate the value function on a batch of state observations.

        This modifies the :attr:`func_approx.params <coax.FuncApprox.params>`
        attribute.

        Parameters
        ----------
        S : ndarray

            A batch of state observations :math:`s`.

        A : ndarray, optional

            A batch of actions :math:`a`. This may be omitted if the action space is discrete.

        Returns
        -------
        Q_sa or Q_s : ndarray

            Depending on whether ``A`` is provided, this either returns a batch of scalars
            representing :math:`q(s,a)\in\mathbb{R}` or a batch of vectors representing
            :math:`q(s,.)\in\mathbb{R}^n`, where :math:`n` is the number of discrete actions.
            Naturally, this only applies for discrete action spaces.

        """
        if A is None:
            Q, _ = self.apply_func_type2(self.params, self.function_state, self.rng, S, False)
        else:
            Q, _ = self.apply_func_type1(self.params, self.function_state, self.rng, S, A, False)
        return Q

    def _init_qtype(self, qtype):
        if qtype not in (1, 2):
            raise ValueError("qtype must be either 1 or 2")
        if qtype == 2 and not self.action_space_is_discrete:
            raise NotImplementedError(
                "type-II q-function is not (yet) implemented for non-discrete "
                "action spaces")
        self._qtype = int(qtype)
        if self._qtype == 1:
            self.COMPONENTS = self.__class__.COMPONENTS_TYPE1
        else:
            self.COMPONENTS = self.__class__.COMPONENTS_TYPE2

    def _init_funcs(self):
        if self.qtype == 1:
            # take both S and A as input
            def apply_func(params, state, rng, S, A, is_training):
                rngs = hk.PRNGSequence(rng)
                body = self.func_approx.apply_funcs['body']
                actn = self.func_approx.apply_funcs['action_preprocessor']
                comb = self.func_approx.apply_funcs['state_action_combiner']
                head = self.func_approx.apply_funcs['head_q1']
                state_new = state.copy()  # shallow copy
                X_s, state_new['body'] = body(
                    params['body'], state['body'], next(rngs), S, is_training)
                X_a = actn(params['action_preprocessor'], next(rngs), A)
                X_sa, state_new['state_action_combiner'] = comb(
                    params['state_action_combiner'], state['state_action_combiner'], next(rngs),
                    X_s, X_a, is_training)
                Q_sa, state_new['head_q1'] = \
                    head(params['head_q1'], state['head_q1'], next(rngs), X_sa, is_training)
                return jnp.squeeze(Q_sa, axis=1), state_new

            self._apply_func = jax.jit(apply_func, static_argnums=5)

        else:
            # take only S as input
            def apply_func(params, state, rng, S, is_training):
                rngs = hk.PRNGSequence(rng)
                body = self.func_approx.apply_funcs['body']
                head = self.func_approx.apply_funcs['head_q2']
                state_new = state.copy()  # shallow copy
                X_s, state_new['body'] = \
                    body(params['body'], state['body'], next(rngs), S, is_training)
                Q_s, state_new['head_q2'] = \
                    head(params['head_q2'], state['head_q2'], next(rngs), X_s, is_training)
                return Q_s, state_new

            self._apply_func = jax.jit(apply_func, static_argnums=4)

        def apply_single_type1_func(params, state, rng, s, a):
            S = single_to_batch(s)
            A = single_to_batch(a)
            Q_sa, _ = self.apply_func_type1(params, state, rng, S, A, False)
            q_sa = batch_to_single(Q_sa)
            return q_sa

        def apply_single_type2_func(params, state, rng, s):
            S = single_to_batch(s)
            Q_s, _ = self.apply_func_type2(params, state, rng, S, False)
            q_s = batch_to_single(Q_s)
            return q_s

        self._apply_single_type1_func = jax.jit(apply_single_type1_func)
        self._apply_single_type2_func = jax.jit(apply_single_type2_func)

    @property
    def apply_func_type1(self):
        r"""

        JIT-compiled function responsible for the forward-pass through the underlying function
        approximator. This function is used by the :attr:`batch_eval` and :attr:`__call__` methods.

        This is the type-I version of the apply-function, regardless of the underlying
        :attr:`qtype`.

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

        A : actions

            A batch of actions.

        is_training : bool

            A flag that indicates whether we are in training mode.

        Returns
        -------
        Q_sa : ndarray

            A batch of state-action values :math:`q(s,a)`.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        """
        if self.qtype == 1:
            return self._apply_func

        if not self.action_space_is_discrete:
            raise ValueError(
                "cannot apply type-II q-function as a type-I q-function if "
                "the action space is non-discrete")

        def q1_func(q2_params, q2_state, rng, S, A, is_training):
            A_onehot = jax.nn.one_hot(A, self.num_actions)
            Q_s, state_new = self._apply_func(q2_params, q2_state, rng, S, is_training)
            Q_sa = jnp.einsum('ij,ij->i', A_onehot, Q_s)
            return Q_sa, state_new

        return q1_func

    @property
    def apply_func_type2(self):
        r"""

        JIT-compiled function responsible for the forward-pass through the underlying function
        approximator. This function is used by the :attr:`batch_eval` and :attr:`__call__` methods.

        This is the type-II version of the apply-function, regardless of the underlying
        :attr:`qtype`.

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
        Q_s : ndarray

            A batch of vector-valued state-action values :math:`q(s,.)`, one for each discrete
            action.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        """
        if self.qtype == 2:
            return self._apply_func

        if not self.action_space_is_discrete:
            raise ValueError(
                "cannot apply type-I q-function as a type-II q-function if "
                "the action space is non-discrete")

        def q2_func(q1_params, q1_state, rng, S, is_training):
            # example: let S = [7, 2, 5, 8] and num_actions = 3, then
            # S_rep = [7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8]  # repeated
            # A_rep = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # tiled
            S_rep = jnp.repeat(S, self.num_actions, axis=0)
            A_rep = jnp.tile(jnp.arange(self.num_actions), S.shape[0])

            # evaluate on replicas => output shape: (batch * num_actions, 1)
            Q_sa_rep, state_new = \
                self._apply_func(q1_params, q1_state, rng, S_rep, A_rep, is_training)
            Q_s = Q_sa_rep.reshape(-1, self.num_actions)  # shape: (batch, num_actions)

            return Q_s, state_new

        return q2_func
