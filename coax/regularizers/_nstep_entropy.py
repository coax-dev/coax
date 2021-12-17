import haiku as hk
import jax
import jax.numpy as jnp

from .._core.base_stochastic_func_type2 import BaseStochasticFuncType2
from ..utils import jit
from ._entropy import EntropyRegularizer


class NStepEntropyRegularizer(EntropyRegularizer):
    r"""

    Policy regularization term based on the n-step entropy of the policy.

    The regularization term is to be added to the loss function:

    .. math::

        \text{loss}(\theta; s,a)\ =\ -J(\theta; s,a) - \beta\,H[\pi_\theta(.|s)]

    where :math:`J(\theta)` is the bare policy objective.

    Parameters
    ----------
    f : stochastic function approximator

        The stochastic function approximator (e.g. :class:`coax.Policy`) to regularize.

    n : tuple(int), list(int), ndarray

        Time indices of the steps (counted from the current state at time `t`)
        to include in the regularization. For example `n = [2, 3]` adds an entropy bonus for the
        policy at the states t + 2 and t + 3 to the objective.

    beta : non-negative float

        The coefficient that determines the strength of the overall regularization term.

    gamma : float between 0 and 1

        The amount by which to discount the entropy bonuses.

    """

    def __init__(self, f, n, beta=0.001, gamma=0.99):
        super().__init__(f)
        if not isinstance(n, (tuple, list, jnp.ndarray)):
            raise TypeError(f"n must be a list, an ndarray or a tuple, got: {type(n)}")
        if len(n) == 0:
            raise ValueError("n cannot be empty")
        self.n = n
        self._n = jnp.array(n)
        self.beta = beta
        self.gamma = gamma
        self._gammas = jnp.take(jnp.power(self.gamma, jnp.arange(self.n[-1] + 1)), self._n)

        def entropy(dist_params, dones):
            valid = self.valid_from_done(dones)
            vf = jax.vmap(lambda p, v, gamma: gamma * self.f.proba_dist.entropy(p) * v)
            return jnp.sum(vf(dist_params, valid, self._gammas), axis=0)

        def function(dist_params, dones, beta):
            assert len(dist_params) == 2
            return -beta * entropy(dist_params, dones)

        def metrics(dist_params, dones, beta):
            assert len(dist_params) == 2
            valid = self.valid_from_done(dones)
            return {
                'EntropyRegularizer/beta': beta,
                'EntropyRegularizer/entropy': jnp.mean(entropy(dist_params, dones) /
                                                       jnp.sum(valid, axis=0))
            }

        self._function = jit(function)
        self._metrics_func = jit(metrics)

    @property
    def batch_eval(self):
        if not hasattr(self, '_batch_eval_func'):
            def batch_eval_func(params, hyperparams, state, rng, transition_batch):
                rngs = hk.PRNGSequence(rng)
                if not isinstance(transition_batch.extra_info, dict):
                    raise TypeError(
                        'TransitionBatch.extra_info has to be a dict containing "states" and' +
                        ' "dones" for the n-step entropy regularization. Make sure to set the' +
                        ' record_extra_info flag in the NStep tracer.')
                if isinstance(self.f, BaseStochasticFuncType2):
                    def f(s_next):
                        return self.f.function(params,
                                               state, next(rngs),
                                               self.f.observation_preprocessor(
                                                   next(rngs), s_next), True)
                    n_states = transition_batch.extra_info['states']
                    dist_params, _ = jax.vmap(f)(jax.tree_util.tree_multimap(
                        lambda *t: jnp.stack(t), *n_states))
                    dist_params = jax.tree_util.tree_map(
                        lambda t: jnp.take(t, self._n, axis=0), dist_params)
                else:
                    raise TypeError(
                        "f must be derived from BaseStochasticFuncType2")
                dones = jnp.stack(transition_batch.extra_info['dones'])
                dones = jnp.take(dones, self._n, axis=0)

                return self.function(dist_params,
                                     dones,
                                     **hyperparams), self.metrics_func(dist_params, dones,
                                                                       **hyperparams)

            self._batch_eval_func = jit(batch_eval_func)

        return self._batch_eval_func

    def valid_from_done(self, dones):
        """
        Generates a mask that filters all time steps after a done signal has been reached.

        Parameters
        ----------
        dones : ndarray

            Array of boolean entries indicating whether the episode has ended.

        Returns
        -------
        valid : ndarray

            Mask that filters all entries after a done=True has been reached.
        """
        valid = jnp.ones_like(dones, dtype=jnp.float32)
        return valid.at[1:].set(1 - jnp.clip(jnp.cumsum(dones[:-1], axis=0), a_max=1))
