import gym
import jax
import jax.numpy as jnp
import haiku as hk
import chex

from ..utils import docstring, is_qfunction, is_stochastic, jit
from ..proba_dists import CategoricalDist
from .base_stochastic_func_type2 import StochasticFuncType2Mixin
from .q import Q


__all__ = (
    'EpsilonGreedy',
    'BoltzmannPolicy',
)


class BaseValueBasedPolicy(StochasticFuncType2Mixin):
    """ Abstract base class for value-based policies. """

    def __init__(self, q):
        if not is_qfunction(q):
            raise TypeError(f"q must be a q-function, got: {type(q)}")

        if not isinstance(q.action_space, gym.spaces.Discrete):
            raise TypeError(f"{self.__class__.__name__} is only well-defined for Discrete actions")

        self.q = q
        self.observation_preprocessor = self.q.observation_preprocessor
        self.action_preprocessor = self.q.action_preprocessor
        self.proba_dist = CategoricalDist(self.q.action_space)

        def Q_s(params, state, rng, S):
            rngs = hk.PRNGSequence(rng)
            if is_stochastic(self.q):
                Q_s = self.q.mean_func_type2(params['q'], state, next(rngs), S)
                Q_s = self.q.proba_dist.postprocess_variate(next(rngs), Q_s, batch_mode=True)
            else:
                Q_s, _ = self.q.function_type2(params['q'], state, next(rngs), S, False)

            chex.assert_rank(Q_s, 2)
            assert Q_s.shape[1] == self.q.action_space.n
            return Q_s

        self._Q_s = jit(Q_s)

    @property
    def rng(self):
        return self.q.rng

    @property
    @docstring(Q.function)
    def function(self):
        return self._function  # this is set downstream (below)

    @property
    @docstring(Q.function_state)
    def function_state(self):
        return self.q.function_state

    @function_state.setter
    def function_state(self, new_function_state):
        self.q.function_state = new_function_state

    def __call__(self, s, return_logp=False):
        r"""

        Sample an action :math:`a\sim\pi_q(.|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        return_logp : bool, optional

            Whether to return the log-propensity :math:`\log\pi_q(a|s)`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        logp : float, optional

            The log-propensity :math:`\log\pi_q(a|s)`. This is only returned if we set
            ``return_logp=True``.

        """
        return super().__call__(s, return_logp=return_logp)

    def mean(self, s):
        r"""

        Get the mean of the distribution :math:`\pi_q(.|s)`.

        Note that if the actions are discrete, this returns the :attr:`mode` instead.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        """
        return super().mean(s)

    def mode(self, s):
        r"""

        Sample a greedy action :math:`a=\arg\max_a\pi_q(a|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        a : action

            A single action :math:`a`.

        """
        return super().mode(s)

    def dist_params(self, s):
        r"""

        Get the conditional distribution parameters of :math:`\pi_q(.|s)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        dist_params : Params

            The distribution parameters of :math:`\pi_q(.|s)`.

        """
        return super().dist_params(s)


class EpsilonGreedy(BaseValueBasedPolicy):
    r"""

    Create an :math:`\epsilon`-greedy policy, given a q-function.

    This policy samples actions :math:`a\sim\pi_q(.|s)` according to the following rule:

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

    """
    def __init__(self, q, epsilon=0.1):
        super().__init__(q)
        self.epsilon = epsilon

        def func(params, state, rng, S, is_training):
            Q_s = self._Q_s(params, state, rng, S)

            A_greedy = (Q_s == Q_s.max(axis=1, keepdims=True)).astype(Q_s.dtype)
            A_greedy /= A_greedy.sum(axis=1, keepdims=True)  # there may be multiple max's (ties)
            A_greedy *= 1 - params['epsilon']                # take away ε from greedy action(s)
            A_greedy += params['epsilon'] / self.q.action_space.n  # spread ε evenly to all actions

            dist_params = {'logits': jnp.log(A_greedy + 1e-15)}
            return dist_params, None  # return dummy function-state

        self._function = jit(func, static_argnums=(4,))

    @property
    @docstring(Q.params)
    def params(self):
        return hk.data_structures.to_immutable_dict({'epsilon': self.epsilon, 'q': self.q.params})

    @params.setter
    def params(self, new_params):
        if jax.tree_structure(new_params) != jax.tree_structure(self.params):
            raise TypeError("new params must have the same structure as old params")
        self.epsilon = new_params['epsilon']
        self.q.params = new_params['q']


class BoltzmannPolicy(BaseValueBasedPolicy):
    r"""

    Derive a Boltzmann policy from a q-function.

    This policy samples actions :math:`a\sim\pi_q(.|s)` according to the following rule:

    .. math::

        p &= \text{softmax}(q(s,.) / \tau) \\
        a &\sim \text{Cat}(p)

    Note that this policy is only well-defined for *discrete* action spaces. Also, it's worth noting
    that if the q-function has a non-trivial value transform :math:`f(.)` (e.g.
    :class:`coax.value_transforms.LogTransform`), we feed in the *transformed* estimate as our
    logits, i.e.

    .. math::

        p = \text{softmax}(f(q(s,.)) / \tau)


    Parameters
    ----------
    q : Q

        A state-action value function.

    temperature : positive float, optional

        The Boltzmann temperature :math:`\tau>0` sets the sharpness of the categorical distribution.
        Picking a small value for :math:`\tau` results in greedy sampling while large values results
        in uniform sampling.

    """
    def __init__(self, q, temperature=0.02):
        super().__init__(q)
        self.temperature = temperature

        def func(params, state, rng, S, is_training):
            Q_s = self._Q_s(params, state, rng, S)
            dist_params = {'logits': Q_s / params['temperature']}
            return dist_params, None  # return dummy function-state

        self._function = jit(func, static_argnums=(4,))

    @property
    @docstring(Q.params)
    def params(self):
        return hk.data_structures.to_immutable_dict(
            {'temperature': self.temperature, 'q': self.q.params})

    @params.setter
    def params(self, new_params):
        if jax.tree_structure(new_params) != jax.tree_structure(self.params):
            raise TypeError("new params must have the same structure as old params")
        self.temperature = new_params['temperature']
        self.q.params = new_params['q']
