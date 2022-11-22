import os
os.environ["MUJOCO_GL"] = "egl"

import coax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import optax

from coax.utils import make_dmc


# the name of this script
name = 'sac'

# the dm_control MDP
env = make_dmc("walker", "walk")
env = coax.wrappers.TrainMonitor(env, name=name)


def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Linear(1024), hk.LayerNorm(-1, create_scale=True, create_offset=True), jax.nn.tanh,
        hk.Linear(1024), jax.nn.relu,
        hk.Linear(onp.prod(env.action_space.shape) * 2),
        hk.Reshape((*env.action_space.shape, 2)),
    ))
    x = seq(S)
    mu, logvar = x[..., 0], x[..., 1]
    return {'mu': mu, 'logvar': logvar}


def func_q(S, A, is_training):
    seq = hk.Sequential((
        hk.Linear(1024), hk.LayerNorm(-1, create_scale=True, create_offset=True), jax.nn.tanh,
        hk.Linear(1024), jax.nn.relu,
        hk.Linear(1), jnp.ravel
    ))
    X = jnp.concatenate((S, A), axis=-1)
    return seq(X)


# main function approximators
pi = coax.Policy(func_pi, env, proba_dist=coax.proba_dists.SquashedNormalDist(
    env.action_space,
    clip_logvar=(-10.0, 4.0),
))
q1 = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)
q2 = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)

# target network
q1_targ = q1.copy()
q2_targ = q2.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.99, record_extra_info=True)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)
policy_regularizer = coax.regularizers.NStepEntropyRegularizer(pi,
                                                               beta=0.2,
                                                               gamma=tracer.gamma,
                                                               n=[tracer.n])

# updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
    q1, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
    loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3),
    policy_regularizer=policy_regularizer)
qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
    q2, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
    loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3),
    policy_regularizer=policy_regularizer)
soft_pg = coax.policy_objectives.SoftPG(pi, [q1_targ, q2_targ], optimizer=optax.adam(
    1e-4), regularizer=policy_regularizer)


# train
while env.T < 1000000:
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 5000:
            transition_batch = buffer.sample(batch_size=256)

            # init metrics dict
            metrics = {}

            # flip a coin to decide which of the q-functions to update
            qlearning = qlearning1 if jax.random.bernoulli(q1.rng) else qlearning2
            metrics.update(qlearning.update(transition_batch))

            # delayed policy updates
            if env.T >= 7500 and env.T % 4 == 0:
                metrics.update(soft_pg.update(transition_batch))

            env.record_metrics(metrics)

            # sync target networks
            q1_targ.soft_update(q1, tau=0.005)
            q2_targ.soft_update(q2, tau=0.005)

        if done or truncated:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000  # round to 10000s
        coax.utils.generate_gif(
            env=env, policy=pi, filepath=f"./data/gifs/{name}/T{T:08d}.gif")
