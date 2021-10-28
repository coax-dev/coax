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

from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from gym.spaces import Space, Discrete

from coax._core.q import Q

from ..utils import safe_sample, default_preprocessor, quantile_func_iqn
from .base_func import ArgsType3, ArgsType4, ExampleData, Inputs, ModelTypes


__all__ = (
    'QuantileQ',
)


class QuantileQ(Q):
    def __init__(self, func, env, quantile_func=None, observation_preprocessor=None,
                 action_preprocessor=None, value_transform=None, random_seed=None):
        self.quantile_func = quantile_func_iqn if quantile_func is None else quantile_func
        super().__init__(func, env, observation_preprocessor=observation_preprocessor,
                         action_preprocessor=action_preprocessor,
                         value_transform=value_transform, random_seed=random_seed)

    def __call__(self, s, a=None, quantiles=None):
        r"""

        Evaluate the given quantiles :math:`tau` of the state-action function on a state
        observation :math:`s` or on a state-action pair :math:`(s, a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        quantiles : quantiles

            The quantiles of the state-action function :math:`\tau`.

        a : action

            A single action :math:`a`.

        Returns
        -------
        q_quantiles_sa or q_quantiles_s : ndarray

            Depending on whether :code:`a` is provided, this either returns a vector representing
            :math:`q_{\tau}(s,a)\in\mathbb{R}^m` or a matrix representing
            :math:`q_{\tau}(s,.)\in\mathbb{R}^{n\times m}`,
            where :math:`n` is the number of discrete actions and :math:`m` is the number of
            quantiles. Naturally, this only applies for discrete action spaces.

        """
        S = self.observation_preprocessor(self.rng, s)
        if a is None:
            if quantiles is None:
                quantiles = self.quantile_func(S=S, rng=self.rng, is_training=False)
            Q, _ = self.function_type4(self.params, self.function_state,
                                       self.rng, S, quantiles, False)
        else:
            A = self.action_preprocessor(self.rng, a)
            if quantiles is None:
                quantiles = self.quantile_func(S=S, A=A, rng=self.rng, is_training=False)
            Q, _ = self.function_type3(self.params, self.function_state,
                                       self.rng, S, A, quantiles, False)
        Q = self.value_transform.inverse_func(Q)
        return onp.asarray(Q[0].mean(axis=-1))

    @property
    def function_type1(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-1 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 1:
            return self.function

        assert isinstance(self.action_space, Discrete)

        def q1_func(q2_params, q2_state, rng, S, A, is_training):
            quantiles = self.quantile_func(S=S, A=A, rng=rng, is_training=is_training)
            Q_Quantiles_sa, state_new = self.function_type3(
                q2_params, q2_state, rng, S, A, quantiles, is_training)
            Q_sa = Q_Quantiles_sa.mean(axis=-1)
            return Q_sa, state_new

        return q1_func

    @property
    def function_type2(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-2 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 2:
            return self.function

        if not isinstance(self.action_space, Discrete):
            raise ValueError(
                "input 'A' is required for type-1 q-function when action space is non-Discrete")

        def q2_func(q1_params, q1_state, rng, S, is_training):
            quantiles = self.quantile_func(S=S, rng=rng, is_training=is_training)
            Q_Quantiles_s, state_new = self.function_type4(q1_params, q1_state, rng, S,
                                                           quantiles, is_training)
            Q_s = Q_Quantiles_s.mean(axis=-1)
            return Q_s, state_new

        return q2_func

    @property
    def function_type3(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-3 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 3:
            return self.function

        assert isinstance(self.action_space, Discrete)

        def q3_func(q2_params, q2_state, rng, S, A, quantiles, is_training):
            assert A.ndim == 2
            assert A.shape[1] == self.action_space.n
            Q_Quantiles_s, state_new = self.function(
                q2_params, q2_state, rng, S, quantiles, is_training)
            Q_Quantiles_sa = jax.vmap(jnp.dot)(A, Q_Quantiles_s)
            return Q_Quantiles_sa, state_new

        return q3_func

    @property
    def function_type4(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-4 function signature, regardless of
        the underlying :attr:`modeltype`.

        """
        if self.modeltype == 4:
            return self.function

        if not isinstance(self.action_space, Discrete):
            raise ValueError(
                "input 'A' is required for type-3 q-function when action space is non-Discrete")

        n = self.action_space.n

        def q4_func(q1_params, q1_state, rng, S, quantiles, is_training):
            rngs = hk.PRNGSequence(rng)
            batch_size = jax.tree_leaves(S)[0].shape[0]
            num_quantiles = jax.tree_leaves(quantiles)[0].shape[-1]

            S_rep = jax.tree_map(lambda x: jnp.repeat(x, n, axis=0), S)
            A_rep = jnp.tile(jnp.arange(n), batch_size)
            A_rep = self.action_preprocessor(next(rngs), A_rep)  # one-hot encoding

            # evaluate on replicas => output shape: (batch * num_actions, num_quantiles)
            Q_Quantiles_sa_rep, state_new = \
                self.function(q1_params, q1_state, next(rngs), S_rep,
                              A_rep, quantiles, is_training)
            # shape: (batch, num_actions, num_quantiles)
            Q_Quantiles_s = Q_Quantiles_sa_rep.reshape(batch_size, n, num_quantiles)

            return Q_Quantiles_s, state_new

        return q4_func

    @property
    def modeltype(self):
        r"""

        Specifier for how the q-function is modeled, i.e.

        .. math::

            (s,a,\tau)  &\mapsto q_{\tau}(s,a)\in\mathbb{R}     &\qquad (\text{modeltype} &= 3) \\
            (s,\tau)    &\mapsto q_{\tau}(s,.)\in\mathbb{R}^n   &\qquad (\text{modeltype} &= 4)

        Note that modeltype=4 is only well-defined if the action space is :class:`Discrete
        <gym.spaces.Discrete>`. Namely, :math:`n` is the number of discrete actions.

        """
        return self._modeltype

    @classmethod
    def example_data(
            cls, env, quantile_func, observation_preprocessor=None, action_preprocessor=None,
            batch_size=1, random_seed=None):

        if not isinstance(env.observation_space, Space):
            raise TypeError(
                "env.observation_space must be derived from gym.Space, "
                f"got: {type(env.observation_space)}")

        if observation_preprocessor is None:
            observation_preprocessor = default_preprocessor(env.observation_space)

        if action_preprocessor is None:
            action_preprocessor = default_preprocessor(env.action_space)

        rnd = onp.random.RandomState(random_seed)
        rngs = hk.PRNGSequence(rnd.randint(jnp.iinfo('int32').max))

        # input: state observations
        S = [safe_sample(env.observation_space, rnd) for _ in range(batch_size)]
        S = [observation_preprocessor(next(rngs), s) for s in S]
        S = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *S)

        # input: actions
        A = [safe_sample(env.action_space, rnd) for _ in range(batch_size)]
        A = [action_preprocessor(next(rngs), a) for a in A]
        A = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *A)

        quantiles = quantile_func(S=S, A=A, rng=next(rngs), is_training=True)

        # output: type3
        q3_data = ExampleData(
            inputs=Inputs(args=ArgsType3(S=S, A=A, quantiles=quantiles,
                          is_training=True), static_argnums=(3,)),
            output=jnp.asarray(rnd.randn(batch_size, quantiles.shape[-1])),
        )

        if not isinstance(env.action_space, Discrete):
            return ModelTypes(type1=None, type2=None, type3=q3_data, type4=None)

        quantiles = quantile_func(S=S, rng=next(rngs), is_training=True)

        # output: type4 (if actions are discrete)
        q4_data = ExampleData(
            inputs=Inputs(args=ArgsType4(S=S, quantiles=quantiles,
                                         is_training=True), static_argnums=(2,)),
            output=jnp.asarray(rnd.randn(batch_size, env.action_space.n, quantiles.shape[-1])),
        )

        return ModelTypes(type1=None, type2=None, type3=q3_data, type4=q4_data)

    def _check_signature(self, func):
        sig_type3 = ('S', 'A', 'quantiles', 'is_training')
        sig_type4 = ('S', 'quantiles', 'is_training')
        sig = tuple(signature(func).parameters)

        if sig not in (sig_type3, sig_type4):
            sig = ', '.join(sig)
            alt = ' or func(S, quantiles, is_training)' if isinstance(
                self.action_space, Discrete) else ''
            raise TypeError(
                f"func has bad signature; expected: func(S, A, quantiles, is_training){alt}, "
                + f"got: func({sig})")

        if sig == sig_type4 and not isinstance(self.action_space, Discrete):
            raise TypeError("type-4 q-functions are only well-defined for Discrete action spaces")

        Env = namedtuple('Env', ('observation_space', 'action_space'))
        example_data_per_modeltype = self.example_data(
            env=Env(self.observation_space, self.action_space),
            quantile_func=self.quantile_func,
            action_preprocessor=self.action_preprocessor,
            batch_size=1,
            random_seed=self.random_seed)

        if sig == sig_type3:
            self._modeltype = 3
            example_data = example_data_per_modeltype.type3
        else:
            self._modeltype = 4
            example_data = example_data_per_modeltype.type4

        return example_data
