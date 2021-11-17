import jax.numpy as jnp

from ..utils import jit
from ._base import Regularizer


class KLDivRegularizer(Regularizer):
    r"""

    Policy regularization term based on the Kullback-Leibler divergence of the policy relative to a
    given set of priors.

    The regularization term is to be added to the loss function:

    .. math::

        \text{loss}(\theta; s,a)\ =\
            -J(\theta; s,a)
            + \beta\,KL[\pi_\theta, \pi_\text{prior}]

    where :math:`J(\theta)` is the bare policy objective. Also, in order to unclutter the notation
    we abbreviated :math:`\pi(.|s)` by :math:`\pi`.

    Parameters
    ----------
    f : stochastic function approximator

        The stochastic function approximator (e.g. :class:`coax.Policy`) to regularize.

    beta : non-negative float

        The coefficient that determines the strength of the overall regularization term.

    priors : pytree with ndarray leaves, optional

        The distribution parameters that correspond to the priors. If left unspecified, we'll use
        :attr:`proba_dist.default_priors`, see e.g. :attr:`NormalDist.default_priors
        <coax.proba_dists.NormalDist.default_priors>`.

    """
    def __init__(self, f, beta=0.001, priors=None):
        super().__init__(f)
        self.beta = beta
        self.priors = self.f.proba_dist.default_priors if priors is None else priors

        def function(dist_params, priors, beta):
            kl_div = self.f.proba_dist.kl_divergence(dist_params, priors)
            return beta * kl_div

        def metrics(dist_params, priors, beta):
            kl_div = self.f.proba_dist.kl_divergence(dist_params, priors)
            return {
                'KLDivRegularizer/beta': beta,
                'KLDivRegularizer/kl_div': jnp.mean(kl_div)}

        self._function = jit(function)
        self._metrics_func = jit(metrics)

    @property
    def hyperparams(self):
        return {'beta': self.beta, 'priors': self.priors}

    @property
    def function(self):
        r"""

        JIT-compiled function that returns the values for the regularization term.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the (conditional) probability distribution.

        beta : non-negative float

            The coefficient that determines the strength of the overall regularization term.

        priors : pytree with ndarray leaves

            The distribution parameters that correspond to the priors.

        """
        return self._function

    @property
    def metrics_func(self):
        r"""

        JIT-compiled function that returns the performance metrics for the regularization term.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the (conditional) probability distribution.

        beta : non-negative float

            The coefficient that determines the strength of the overall regularization term.

        priors : pytree with ndarray leaves

            The distribution parameters that correspond to the priors.

        """
        return self._metrics_func
