import haiku as hk
import jax
import jax.numpy as jnp
from gym.spaces import Discrete

from ..utils import (batch_to_single, is_stochastic, single_to_batch,
                     stack_trees)
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
            A_next_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))

            # compute A_next from q_i
            for q_i, params_i, state_i in qs:
                S_next = q_i.observation_preprocessor(next(rngs), transition_batch.S_next)
                if is_stochastic(q_i):
                    Q_s_next = q_i.mean_func_type2(params_i, state_i, next(rngs), S_next)
                    Q_s_next = q_i.proba_dist.postprocess_variate(
                        next(rngs), Q_s_next, batch_mode=True)
                else:
                    Q_s_next, _ = q_i.function_type2(params_i, state_i, next(rngs), S_next, False)
                assert Q_s_next.ndim == 2, f"bad shape: {Q_s_next.shape}"
                A_next = (Q_s_next == Q_s_next.max(axis=1, keepdims=True)).astype(Q_s_next.dtype)
                A_next /= A_next.sum(axis=1, keepdims=True)  # there may be ties

                # evaluate on q_j
                for q_j, params_j, state_j in qs:
                    S_next = q_j.observation_preprocessor(next(rngs), transition_batch.S_next)
                    if is_stochastic(q_j):
                        Q_sa_next = q_j.mean_func_type1(
                            params_j, state_j, next(rngs), S_next, A_next)
                        Q_sa_next = q_j.proba_dist.postprocess_variate(
                            next(rngs), Q_sa_next, batch_mode=True)
                    else:
                        Q_sa_next, _ = q_j.function_type1(
                            params_j, state_j, next(rngs), S_next, A_next, False)
                    assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
                    f_inv = q_j.value_transform.inverse_func
                    Q_sa_next_list.append(f_inv(Q_sa_next))
                    A_next_list.append(A_next)

        else:
            Q_sa_next_list = []
            A_next_list = []
            qs = list(zip(self.q_targ_list, target_params['q_targ'], target_state['q_targ']))
            pis = list(zip(self.pi_targ_list, target_params['pi_targ'], target_state['pi_targ']))

            # compute A_next from pi_i
            for pi_i, params_i, state_i in pis:
                S_next = pi_i.observation_preprocessor(next(rngs), transition_batch.S_next)
                dist_params, _ = pi_i.function(params_i, state_i, next(rngs), S_next, False)
                A_next = pi_i.proba_dist.sample(dist_params, next(rngs))  # sample instead of mode

                # evaluate on q_j
                for q_j, params_j, state_j in qs:
                    S_next = q_j.observation_preprocessor(next(rngs), transition_batch.S_next)
                    if is_stochastic(q_j):
                        Q_sa_next = q_j.mean_func_type1(
                            params_j, state_j, next(rngs), S_next, A_next)
                        Q_sa_next = q_j.proba_dist.postprocess_variate(
                            next(rngs), Q_sa_next, batch_mode=True)
                    else:
                        Q_sa_next, _ = q_j.function_type1(
                            params_j, state_j, next(rngs), S_next, A_next, False)
                    assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
                    f_inv = q_j.value_transform.inverse_func
                    Q_sa_next_list.append(f_inv(Q_sa_next))
                    A_next_list.append(A_next)

        # take the min to mitigate over-estimation
        A_next_list = jnp.stack(A_next_list, axis=1)
        Q_sa_next_list = jnp.stack(Q_sa_next_list, axis=-1)
        assert Q_sa_next_list.ndim == 2, f"bad shape: {Q_sa_next_list.shape}"

        if is_stochastic(self.q):
            Q_sa_next_argmin = jnp.argmin(Q_sa_next_list, axis=-1)
            Q_sa_next_argmin_q = Q_sa_next_argmin % len(self.q_targ_list)

            def target_dist_params(A_next_idx, q_targ_idx, p, s, t, A_next_list):
                return self._get_target_dist_params(batch_to_single(p, q_targ_idx),
                                                    batch_to_single(s, q_targ_idx),
                                                    next(rngs),
                                                    single_to_batch(t),
                                                    single_to_batch(batch_to_single(A_next_list,
                                                                                    A_next_idx)))

            def tile_parameters(params, state, reps):
                return jax.tree_util.tree_map(lambda t: jnp.tile(t, [reps, *([1] * (t.ndim - 1))]),
                                              stack_trees(params, state))
            # stack and tile q-function params to select the argmin for the target dist params
            tiled_target_params, tiled_target_state = tile_parameters(
                target_params['q_targ'], target_state['q_targ'], reps=len(self.q_targ_list))

            vtarget_dist_params = jax.vmap(target_dist_params, in_axes=(0, 0, None, None, 0, 0))
            dist_params = vtarget_dist_params(
                Q_sa_next_argmin,
                Q_sa_next_argmin_q,
                tiled_target_params,
                tiled_target_state,
                transition_batch,
                A_next_list)
            # unwrap dist params computed for single batches
            return jax.tree_util.tree_map(lambda t: jnp.squeeze(t, axis=1), dist_params)

        Q_sa_next = jnp.min(Q_sa_next_list, axis=-1)
        assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
        f = self.q.value_transform.transform_func
        return f(transition_batch.Rn + transition_batch.In * Q_sa_next)
