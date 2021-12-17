import haiku as hk
from jax.scipy.special import logsumexp
from gym.spaces import Discrete

from ..utils import is_stochastic
from ._base import BaseTDLearningQ


class SoftQLearning(BaseTDLearningQ):
    r"""

    TD-learning with soft Q-learning updates. The :math:`n`-step bootstrapped target is constructed
    as:

    .. math::

        G^{(n)}_t\ =\ R^{(n)}_t
            + I^{(n)}_t\,\tau\log\sum_{a'}\exp\left(q_\text{targ}(S_{t+n}, a') / \tau\right)

    where

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

    temperature : float, optional

        The Boltzmann temperature :math:`\tau>0`.

    """
    def __init__(
            self, q, q_targ=None, optimizer=None, loss_function=None, policy_regularizer=None,
            temperature=1.0):

        if not isinstance(q.action_space, Discrete):
            raise NotImplementedError(
                f"{self.__class__.__name__} class is only implemented for discrete actions spaces")
        if is_stochastic(q):
            raise NotImplementedError(f"{type(self).__name__} is not yet implement for StochasticQ")

        self.temperature = temperature
        super().__init__(
            q=q,
            q_targ=q_targ,
            optimizer=optimizer,
            loss_function=loss_function,
            policy_regularizer=policy_regularizer)

    def target_func(self, target_params, target_state, rng, transition_batch):
        rngs = hk.PRNGSequence(rng)
        params, state = target_params['q_targ'], target_state['q_targ']
        S_next = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S_next)
        Q_s_next, _ = self.q_targ.function_type2(params, state, next(rngs), S_next, False)
        assert Q_s_next.ndim == 2
        assert Q_s_next.shape[1] == self.q.action_space.n
        Q_sa_next = self.temperature * logsumexp(Q_s_next / self.temperature, axis=-1)
        assert Q_sa_next.ndim == 1

        f, f_inv = self.q.value_transform.transform_func, self.q_targ.value_transform.inverse_func
        return f(transition_batch.Rn + transition_batch.In * f_inv(Q_sa_next))
