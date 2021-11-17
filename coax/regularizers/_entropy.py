import jax.numpy as jnp

from ..utils import jit
from ._base import Regularizer


class EntropyRegularizer(Regularizer):
    r"""

    Policy regularization term based on the entropy of the policy.

    The regularization term is to be added to the loss function:

    .. math::

        \text{loss}(\theta; s,a)\ =\ -J(\theta; s,a) - \beta\,H[\pi_\theta(.|s)]

    where :math:`J(\theta)` is the bare policy objective.

    Parameters
    ----------
    f : stochastic function approximator

        The stochastic function approximator (e.g. :class:`coax.Policy`) to regularize.

    beta : non-negative float

        The coefficient that determines the strength of the overall regularization term.

    """
    def __init__(self, f, beta=0.001):
        super().__init__(f)
        self.beta = beta

        def function(dist_params, beta):
            entropy = self.f.proba_dist.entropy(dist_params)
            return -beta * entropy

        def metrics(dist_params, beta):
            entropy = self.f.proba_dist.entropy(dist_params)
            return {
                'EntropyRegularizer/beta': beta,
                'EntropyRegularizer/entropy': jnp.mean(entropy)}

        self._function = jit(function)
        self._metrics_func = jit(metrics)

    @property
    def hyperparams(self):
        return {'beta': self.beta}

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

        """
        return self._metrics_func
