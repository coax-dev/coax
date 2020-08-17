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

import warnings
from inspect import signature
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as onp
from gym.spaces import Space, Discrete

from ..utils import single_to_batch, safe_sample
from ..proba_dists import ProbaDist
from .base_func import BaseFunc, ExampleData, Inputs


__all__ = (
    'Q',
)


QTypes = namedtuple('QTypes', ('type1', 'type2'))
ArgsType1 = namedtuple('Args', ('S', 'A', 'is_training'))
ArgsType2 = namedtuple('Args', ('S', 'is_training'))


class Q(BaseFunc):
    r"""

    A state-action value function :math:`q_\theta(s,a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass. The function signature must be the
        same as the example below.

    observation_space : gym.Space

        The observation space of the environment. This is used to generate example input for
        initializing :attr:`params`.

    action_space : gym.Space

        The action space of the environment. This may be used to generate example input for
        initializing :attr:`params` or to validate the output structure.

    action_preprocessor : function, optional

        Turns a single actions into a batch of actions that are compatible with the corresponding
        probability distribution. If left unspecified, this defaults to:

        .. code:: python

            action_preprocessor = ProbaDist(action_space).preprocess_variate

        See also :attr:`coax.proba_dists.ProbaDist.preprocess_variate`.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, observation_space, action_space,
            action_preprocessor=None, random_seed=None):

        self.action_preprocessor = \
            action_preprocessor if action_preprocessor is not None \
            else ProbaDist(action_space).preprocess_variate

        super().__init__(
            func,
            observation_space=observation_space,
            action_space=action_space,
            random_seed=random_seed)

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
        S = single_to_batch(s)
        if a is None:
            Q, _ = self.function_type2(self.params, self.function_state, self.rng, S, False)
        else:
            A = self.action_preprocessor(a)
            Q, _ = self.function_type1(self.params, self.function_state, self.rng, S, A, False)
        return onp.asarray(Q[0])

    @property
    def function_type1(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-1 function signature, regardless of
        the underlying :attr:`qtype`.

        """
        if self.qtype == 1:
            return self.function

        assert isinstance(self.action_space, Discrete)

        def q1_func(q2_params, q2_state, rng, S, A, is_training):
            assert A.ndim == 2
            assert A.shape[1] == self.action_space.n
            Q_s, state_new = self.function(q2_params, q2_state, rng, S, is_training)
            Q_sa = jnp.einsum('ij,ij->i', A, Q_s)
            return Q_sa, state_new

        return q1_func

    @property
    def function_type2(self):
        r"""

        Same as :attr:`function`, except that it ensures a type-2 function signature, regardless of
        the underlying :attr:`qtype`.

        """
        if self.qtype == 2:
            return self.function

        if not isinstance(self.action_space, Discrete):
            raise ValueError(
                "input 'A' is required for type-1 q-function when action space is non-Discrete")

        n = self.action_space.n

        def q2_func(q1_params, q1_state, rng, S, is_training):
            # example: let S = [7, 2, 5, 8] and num_actions = 3, then
            # S_rep = [7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8]  # repeated
            # A_rep = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # tiled
            S_rep = jnp.repeat(S, n, axis=0)
            A_rep = jnp.tile(jnp.arange(n), S.shape[0])
            A_rep = self.action_preprocessor(A_rep)

            # evaluate on replicas => output shape: (batch * num_actions, 1)
            Q_sa_rep, state_new = self.function(q1_params, q1_state, rng, S_rep, A_rep, is_training)
            Q_s = Q_sa_rep.reshape(-1, n)  # shape: (batch, num_actions)

            return Q_s, state_new

        return q2_func

    @property
    def qtype(self):
        r"""

        Specifier for how the q-function is modeled, i.e.

        .. math::

            (s,a)   &\mapsto q(s,a)\in\mathbb{R}    &\qquad (\text{qtype} &= 1) \\
            s       &\mapsto q(s,.)\in\mathbb{R}^n  &\qquad (\text{qtype} &= 2)

        Note that qtype=2 is only well-defined if the action space is :class:`Discrete
        <gym.spaces.Discrete>`. Namely, :math:`n` is the number of discrete actions.

        """
        return self._qtype

    @classmethod
    def example_data(
            cls, observation_space, action_space,
            action_preprocessor=None, batch_size=1, random_seed=None):

        if not isinstance(observation_space, Space):
            raise TypeError(
                f"observation_space must be derived from gym.Space, got: {type(observation_space)}")

        rnd = onp.random.RandomState(random_seed)

        # input: state observations
        S = [safe_sample(observation_space, rnd) for _ in range(batch_size)]
        S = jax.tree_multimap(lambda *x: jnp.stack(x, axis=0), *S)

        # input: actions
        A = jax.tree_multimap(
            lambda *x: jnp.stack(x, axis=0),
            *(safe_sample(action_space, rnd) for _ in range(batch_size)))
        try:
            if action_preprocessor is None:
                action_preprocessor = ProbaDist(action_space).preprocess_variate
            A = action_preprocessor(A)
        except Exception as e:
            warnings.warn(f"preprocessing failed for actions A; caught exception: {e}")

        q1_data = ExampleData(
            inputs=Inputs(args=ArgsType1(S=S, A=A, is_training=True), static_argnums=(2,)),
            output=jnp.asarray(rnd.randn(batch_size)),
        )
        q2_data = None
        if isinstance(action_space, Discrete):
            q2_data = ExampleData(
                inputs=Inputs(args=ArgsType2(S=S, is_training=True), static_argnums=(1,)),
                output=jnp.asarray(rnd.randn(batch_size, action_space.n)),
            )

        return QTypes(type1=q1_data, type2=q2_data)

    def _check_signature(self, func):
        sig_type1 = ('S', 'A', 'is_training')
        sig_type2 = ('S', 'is_training')
        sig = tuple(signature(func).parameters)

        if sig not in (sig_type1, sig_type2):
            sig = ', '.join(sig)
            alt = ' or func(S, is_training)' if isinstance(self.action_space, Discrete) else ''
            raise TypeError(
                f"func has bad signature; expected: func(S, A, is_training){alt}, got: func({sig})")

        if sig == sig_type2 and not isinstance(self.action_space, Discrete):
            raise TypeError("type-2 q-functions are only well-defined for Discrete action spaces")

        example_data_per_qtype = self.example_data(
            observation_space=self.observation_space,
            action_space=self.action_space,
            action_preprocessor=self.action_preprocessor,
            batch_size=1,
            random_seed=self.random_seed)

        if sig == sig_type1:
            self._qtype = 1
            example_data = example_data_per_qtype.type1
        else:
            self._qtype = 2
            example_data = example_data_per_qtype.type2

        return example_data

    def _check_output(self, actual, expected):
        if not isinstance(actual, jnp.ndarray):
            class_name = actual.__class__.__name__
            raise TypeError(f"func has bad return type; expected jnp.ndarray, got {class_name}")

        if not jnp.issubdtype(actual.dtype, jnp.floating):
            raise TypeError(
                "func has bad return dtype; expected a subdtype of jnp.floating, "
                f"got dtype={actual.dtype}")

        if actual.shape != expected.shape:
            raise TypeError(
                f"func has bad return shape, expected: {expected.shape}, got: {actual.shape}")
