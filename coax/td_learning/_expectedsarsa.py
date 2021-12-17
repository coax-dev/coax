import gym
import jax
import haiku as hk

from ..utils import is_stochastic
from ._base import BaseTDLearningQWithTargetPolicy


class ExpectedSarsa(BaseTDLearningQWithTargetPolicy):
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

    optimizer : optax optimizer, optional

        An optax-style optimizer. The default optimizer is :func:`optax.adam(1e-3)
        <optax.adam>`.

    loss_function : callable, optional

        The loss function that will be used to regress to the (bootstrapped) target. The loss
        function is expected to be of the form:

        .. math::

            L(y_\text{true}, y_\text{pred}, w)\in\mathbb{R}

        where :math:`w>0` are sample weights. If left unspecified, this defaults to
        :func:`coax.value_losses.huber`. Check out the :mod:`coax.value_losses` module for other
        predefined loss functions.

    policy_regularizer : Regularizer, optional

        If provided, this policy regularizer is added to the TD-target. A typical example is to use
        an :class:`coax.regularizers.EntropyRegularizer`, which adds the policy entropy to
        the target. In this case, we minimize the following loss shifted by the entropy term:

        .. math::

            L(y_\text{true} + \beta\,H[\pi], y_\text{pred})

        Note that the coefficient :math:`\beta` plays the role of the temperature in SAC-style
        agents.

    """
    def __init__(
            self, q, pi_targ, q_targ=None, optimizer=None,
            loss_function=None, policy_regularizer=None):

        if not isinstance(q.action_space, gym.spaces.Discrete):
            raise NotImplementedError(
                f"{self.__class__.__name__} class is only implemented for discrete actions spaces")
        if pi_targ is None:
            raise TypeError("pi_targ must be provided")

        super().__init__(
            q=q,
            pi_targ=pi_targ,
            q_targ=q_targ,
            optimizer=optimizer,
            loss_function=loss_function,
            policy_regularizer=policy_regularizer)

    def target_func(self, target_params, target_state, rng, transition_batch):
        rngs = hk.PRNGSequence(rng)

        # action propensities
        params, state = target_params['pi_targ'], target_state['pi_targ']
        S_next = self.pi_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
        dist_params, _ = self.pi_targ.function(params, state, next(rngs), S_next, False)
        A_next = jax.nn.softmax(dist_params['logits'], axis=-1)  # only works for Discrete actions

        # evaluate on q_targ
        params, state = target_params['q_targ'], target_state['q_targ']
        S_next = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S_next)

        if is_stochastic(self.q):
            return self._get_target_dist_params(params, state, next(rngs), transition_batch, A_next)

        Q_sa_next, _ = self.q_targ.function_type1(params, state, next(rngs), S_next, A_next, False)
        f, f_inv = self.q.value_transform.transform_func, self.q_targ.value_transform.inverse_func
        return f(transition_batch.Rn + transition_batch.In * f_inv(Q_sa_next))
