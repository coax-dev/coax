import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental.optix import adam
from ray.rllib.env.atari_wrappers import wrap_deepmind


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/ddpg"
gifs_filepath = "./data/gifs/ddpg/T{:08d}.gif"


# env with preprocessing
env = gym.make('PongNoFrameskip-v4')  # wrap_deepmind will do frame skipping
env = wrap_deepmind(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir=tensorboard_dir)


def shared(S, is_training):
    M = coax.utils.diff_transform_matrix(num_frames=S.shape[-1])
    seq = hk.Sequential([  # S.shape = [batch, h, w, num_stack]
        lambda x: jnp.dot(x / 255, M),  # [b, h, w, n]
        hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
        hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
        hk.Flatten(),
    ])
    return seq(S)


def func_pi(S, is_training):
    logits = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros),
    ))
    X = shared(S, is_training)
    return {'logits': logits(X)}


def func_q(S, A, is_training):
    value = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    X = shared(S, is_training)
    assert A.ndim == 2 and A.shape[1] == env.action_space.n, "actions must be one-hot encoded"
    return value(jax.vmap(jnp.kron)(X, A))


# function approximators
pi = coax.Policy(func_pi, env.observation_space, env.action_space)
q = coax.Q(func_q, env.observation_space, env.action_space)

# target networks
pi_targ = pi.copy()
q_targ = q.copy()

# policy regularizer (avoid premature exploitation)
kl_div = coax.policy_regularizers.KLDivRegularizer(pi, beta=0.001)

# updaters
qlearning = coax.td_learning.QLearningMode(q, pi_targ, q_targ, optimizer=adam(3e-4))
determ_pg = coax.policy_objectives.DeterministicPG(pi, q, regularizer=kl_div, optimizer=adam(3e-4))

# reward tracer and replay buffer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


while env.T < 3000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) > 50000:  # buffer warm-up
            transition_batch = buffer.sample(batch_size=32)
            env.record_metrics(determ_pg.update(transition_batch))
            env.record_metrics(qlearning.update(transition_batch))

        if env.period('target_model_sync', T_period=10000):
            pi_targ.soft_update(pi, tau=1)
            q_targ.soft_update(q, tau=1)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 50000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(
            env=env, policy=pi.greedy, resize_to=(320, 420),
            filepath=gifs_filepath.format(T))
