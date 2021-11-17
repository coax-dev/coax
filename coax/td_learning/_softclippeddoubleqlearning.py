import jax.numpy as jnp
import haiku as hk
from gym.spaces import Discrete

from ._clippeddoubleqlearning import ClippedDoubleQLearning


class SoftClippedDoubleQLearning(ClippedDoubleQLearning):

    def target_func(self, target_params, target_state, rng, transition_batch):
        """
        This does almost the same as `ClippedDoubleQLearning.target_func` except that
        the action for the next state is sampled instead of taking the mode.
        """
        rngs = hk.PRNGSequence(rng)

        # collect list of q-values
        if isinstance(self.q.action_space, Discrete):
            Q_sa_next_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))

            # compute A_next from q_i
            for q_i, params_i, state_i in qs:
                S_next = q_i.observation_preprocessor(next(rngs), transition_batch.S_next)
                Q_s_next, _ = q_i.function_type2(params_i, state_i, next(rngs), S_next, False)
                assert Q_s_next.ndim == 2, f"bad shape: {Q_s_next.shape}"
                A_next = (Q_s_next == Q_s_next.max(axis=1, keepdims=True)).astype(Q_s_next.dtype)
                A_next /= A_next.sum(axis=1, keepdims=True)  # there may be ties

                # evaluate on q_j
                for q_j, params_j, state_j in qs:
                    S_next = q_j.observation_preprocessor(next(rngs), transition_batch.S_next)
                    Q_sa_next, _ = q_j.function_type1(
                        params_j, state_j, next(rngs), S_next, A_next, False)
                    assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
                    f_inv = q_j.value_transform.inverse_func
                    Q_sa_next_list.append(f_inv(Q_sa_next))

        else:
            Q_sa_next_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))
            pis = list(zip(self.pi_targ_list, target_params['pi_targ'], target_state['pi_targ']))

            # compute A_next from pi_i
            for pi_i, params_i, state_i in pis:
                S_next = pi_i.observation_preprocessor(next(rngs), transition_batch.S_next)
                dist_params, _ = pi_i.function(params_i, state_i, next(rngs), S_next, False)
                A_next = pi_i.proba_dist.sample(dist_params, next(rngs))

                # evaluate on q_j
                for q_j, params_j, state_j in qs:
                    S_next = q_j.observation_preprocessor(next(rngs), transition_batch.S_next)
                    Q_sa_next, _ = q_j.function_type1(
                        params_j, state_j, next(rngs), S_next, A_next, False)
                    assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
                    f_inv = q_j.value_transform.inverse_func
                    Q_sa_next_list.append(f_inv(Q_sa_next))

        # take the min to mitigate over-estimation
        Q_sa_next_list = jnp.stack(Q_sa_next_list, axis=-1)
        assert Q_sa_next_list.ndim == 2, f"bad shape: {Q_sa_next_list.shape}"
        Q_sa_next = jnp.min(Q_sa_next_list, axis=-1)

        assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
        f = self.q.value_transform.transform_func
        return f(transition_batch.Rn + transition_batch.In * Q_sa_next)
