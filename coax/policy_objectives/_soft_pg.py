import jax.numpy as jnp
import haiku as hk
import chex


from ._base import PolicyObjective
from ..utils import is_qfunction, is_stochastic


class SoftPG(PolicyObjective):

    def __init__(self, pi, q_targ_list, optimizer=None, regularizer=None):
        super().__init__(pi, optimizer=optimizer, regularizer=regularizer)
        self._check_input_lists(q_targ_list)
        self.q_targ_list = q_targ_list

    @property
    def hyperparams(self):
        return hk.data_structures.to_immutable_dict({
            'regularizer': getattr(self.regularizer, 'hyperparams', {}),
            'q': {'params': [q_targ.params for q_targ in self.q_targ_list],
                  'function_state': [q_targ.function_state for q_targ in self.q_targ_list]}})

    def objective_func(self, params, state, hyperparams, rng, transition_batch, Adv):
        rngs = hk.PRNGSequence(rng)

        # get distribution params from function approximator
        S = self.pi.observation_preprocessor(next(rngs), transition_batch.S)
        dist_params, state_new = self.pi.function(params, state, next(rngs), S, True)
        A = self.pi.proba_dist.sample(dist_params, next(rngs))
        log_pi = self.pi.proba_dist.log_proba(dist_params, A)

        Q_sa_list = []
        qs = list(zip(self.q_targ_list, hyperparams['q']
                  ['params'], hyperparams['q']['function_state']))

        for q_targ, params_q, state_q in qs:
            # compute objective: q(s, a)
            S = q_targ.observation_preprocessor(next(rngs), transition_batch.S)
            if is_stochastic(q_targ):
                dist_params_q, _ = q_targ.function_type1(params_q, state_q, rng, S, A, True)
                Q = q_targ.proba_dist.mean(dist_params_q)
                Q = q_targ.proba_dist.postprocess_variate(next(rngs), Q, batch_mode=True)
            else:
                Q, _ = q_targ.function_type1(params_q, state_q, next(rngs), S, A, True)
            Q_sa_list.append(Q)
        # take the min to mitigate over-estimation
        Q_sa_next_list = jnp.stack(Q_sa_list, axis=-1)
        assert Q_sa_next_list.ndim == 2, f"bad shape: {Q_sa_next_list.shape}"
        Q = jnp.min(Q_sa_next_list, axis=-1)
        assert Q.ndim == 1, f"bad shape: {Q.shape}"

        # clip importance weights to reduce variance
        W = jnp.clip(transition_batch.W, 0.1, 10.)

        # the objective
        chex.assert_equal_shape([W, Q])
        chex.assert_rank([W, Q], 1)
        objective = W * Q

        return jnp.mean(objective), (dist_params, log_pi, state_new)

    def _check_input_lists(self, q_targ_list):
        # check input: q_targ_list
        if not isinstance(q_targ_list, (tuple, list)):
            raise TypeError(f"q_targ_list must be a list or a tuple, got: {type(q_targ_list)}")
        if not q_targ_list:
            raise ValueError("q_targ_list cannot be empty")
        for q_targ in q_targ_list:
            if not is_qfunction(q_targ):
                raise TypeError(f"all q_targ in q_targ_list must be a coax.Q, got: {type(q_targ)}")

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
