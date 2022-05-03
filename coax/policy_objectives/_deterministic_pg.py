import warnings

import jax.numpy as jnp
import haiku as hk
import chex

from ..utils import check_preprocessors, is_qfunction, is_stochastic
from ._base import PolicyObjective


class DeterministicPG(PolicyObjective):
    r"""
    A deterministic policy-gradient objective, a.k.a. DDPG-style objective. See
    :doc:`spinup:algorithms/ddpg` and references therein for more details.

    .. math::

        J(\theta; s,a)\ =\ q_\text{targ}(s, a_\theta(s))

    Here :math:`a_\theta(s)` is the *mode* of the underlying conditional
    probability distribution :math:`\pi_\theta(.|s)`. See e.g. the :attr:`mode`
    method of :class:`coax.proba_dists.NormalDist`. In other words, we evaluate
    the policy according to the current estimate of its best-case performance.

    This objective has the property that it explicitly maximizes the q-value.

    The full policy loss is constructed as:

    .. math::

        \text{loss}(\theta; s,a)\ =\
            -J(\theta; s,a)
            - \beta_\text{ent}\,H[\pi_\theta]
            + \beta_\text{kl-div}\,
                KL[\pi_{\theta_\text{prior}}, \pi_\theta]

    N.B. in order to unclutter the notation we abbreviated
    :math:`\pi(.|s)` by :math:`\pi`.

    Parameters
    ----------
    pi : Policy

        The parametrized policy :math:`\pi_\theta(a|s)`.

    q_targ : Q

        The target state-action value function :math:`q_\text{targ}(s,a)`.

    optimizer : optax optimizer, optional

        An optax-style optimizer. The default optimizer is :func:`optax.adam(1e-3)
        <optax.adam>`.

    regularizer : Regularizer, optional

        A policy regularizer, see :mod:`coax.regularizers`.

    """
    REQUIRES_PROPENSITIES = False

    def __init__(self, pi, q_targ, optimizer=None, regularizer=None):
        if not is_qfunction(q_targ):
            raise TypeError(f"q must be a q-function, got: {type(q_targ)}")
        if q_targ.modeltype != 1:
            raise TypeError("q must be a type-1 q-function")

        super().__init__(pi=pi, optimizer=optimizer, regularizer=regularizer)
        self.q_targ = q_targ

        if not check_preprocessors(
                self.pi.action_space,
                self.q_targ.action_preprocessor,
                self.pi.proba_dist.preprocess_variate):
            warnings.warn(
                "it seems that q_targ.action_preprocessor does not match "
                "pi.proba_dist.preprocess_variate; please instantiate your q-function using "
                "q = coax.Q(..., action_preprocessor=pi.proba_dist.preprocess_variate) to ensure "
                "that the preprocessors match")

    @property
    def hyperparams(self):
        return hk.data_structures.to_immutable_dict({
            'regularizer': getattr(self.regularizer, 'hyperparams', {}),
            'q': {'params': self.q_targ.params, 'function_state': self.q_targ.function_state}})

    def objective_func(self, params, state, hyperparams, rng, transition_batch, Adv):
        rngs = hk.PRNGSequence(rng)

        # get distribution params from function approximator
        S = self.pi.observation_preprocessor(next(rngs), transition_batch.S)
        dist_params, state_new = self.pi.function(params, state, next(rngs), S, True)

        # compute objective: q(s, a_greedy)
        S = self.q_targ.observation_preprocessor(next(rngs), transition_batch.S)
        A = self.pi.proba_dist.mode(dist_params)
        log_pi = self.pi.proba_dist.log_proba(dist_params, A)
        params_q, state_q = hyperparams['q']['params'], hyperparams['q']['function_state']
        if is_stochastic(self.q_targ):
            dist_params_q, _ = self.q_targ.function_type1(params_q, state_q, rng, S, A, True)
            Q = self.q_targ.proba_dist.mean(dist_params_q)
            Q = self.q_targ.proba_dist.postprocess_variate(next(rngs), Q, batch_mode=True)
        else:
            Q, _ = self.q_targ.function_type1(params_q, state_q, next(rngs), S, A, True)

        # clip importance weights to reduce variance
        W = jnp.clip(transition_batch.W, 0.1, 10.)

        # the objective
        chex.assert_equal_shape([W, Q])
        chex.assert_rank([W, Q], 1)
        objective = W * Q

        return jnp.mean(objective), (dist_params, log_pi, state_new)

    def update(self, transition_batch, Adv=None):
        r"""

        Update the model parameters (weights) of the underlying function approximator.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Adv : ndarray, ignored

            This input is ignored; it is included for consistency with other policy objectives.

        Returns
        -------
        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return super().update(transition_batch, None)

    def grads_and_metrics(self, transition_batch, Adv=None):
        r"""

        Compute the gradients associated with a batch of transitions with
        corresponding advantages.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A batch of transitions.

        Adv : ndarray, ignored

            This input is ignored; it is included for consistency with other policy objectives.

        Returns
        -------
        grads : pytree with ndarray leaves

            A batch of gradients.

        function_state : pytree

            The internal state of the forward-pass function. See :attr:`Policy.function_state
            <coax.Policy.function_state>` and :func:`haiku.transform_with_state` for more details.

        metrics : dict of scalar ndarrays

            The structure of the metrics dict is ``{name: score}``.

        """
        return super().grads_and_metrics(transition_batch, None)
