import jax.numpy as jnp
import haiku as hk
import chex

from ._base import PolicyObjective


class VanillaPG(PolicyObjective):
    r"""
    A vanilla policy-gradient objective, a.k.a. REINFORCE-style objective.

    .. math::

        J(\theta; s,a)\ =\ \mathcal{A}(s,a)\,\log\pi_\theta(a|s)

    This objective has the property that its gradient with respect to
    :math:`\theta` yields the REINFORCE-style policy gradient.

    Parameters
    ----------
    pi : Policy

        The parametrized policy :math:`\pi_\theta(a|s)`.

    optimizer : optax optimizer, optional

        An optax-style optimizer. The default optimizer is :func:`optax.adam(1e-3)
        <optax.adam>`.

    regularizer : Regularizer, optional

        A policy regularizer, see :mod:`coax.regularizers`.

    """
    REQUIRES_PROPENSITIES = False

    def objective_func(self, params, state, hyperparams, rng, transition_batch, Adv):
        rngs = hk.PRNGSequence(rng)

        # get distribution params from function approximator
        S = self.pi.observation_preprocessor(next(rngs), transition_batch.S)
        dist_params, state_new = self.pi.function(params, state, next(rngs), S, True)

        # compute REINFORCE-style objective
        A = self.pi.proba_dist.preprocess_variate(next(rngs), transition_batch.A)
        log_pi = self.pi.proba_dist.log_proba(dist_params, A)

        # clip importance weights to reduce variance
        W = jnp.clip(transition_batch.W, 0.1, 10.)

        # some consistency checks
        chex.assert_equal_shape([W, Adv, log_pi])
        chex.assert_rank([W, Adv, log_pi], 1)
        objective = W * Adv * log_pi

        # also pass auxiliary data to avoid multiple forward passes
        return jnp.mean(objective), (dist_params, log_pi, state_new)
