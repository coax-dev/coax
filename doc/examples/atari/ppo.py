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
tensorboard_dir = "./data/tensorboard/ppo"
gifs_filepath = "./data/gifs/ppo/T{:08d}.gif"


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


def func_v(S, is_training):
    value = hk.Sequential((
        hk.Linear(256), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    X = shared(S, is_training)
    return value(X)


# function approximators
pi = coax.Policy(func_pi, env.observation_space, env.action_space)
v = coax.V(func_v, env.observation_space)

# target networks
pi_behavior = pi.copy()
v_targ = v.copy()

# policy regularizer (avoid premature exploitation)
kl_div = coax.policy_regularizers.KLDivRegularizer(pi, beta=0.001)

# updaters
simpletd = coax.td_learning.SimpleTD(v, v_targ, optimizer=adam(3e-4))
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=kl_div, optimizer=adam(3e-4))

# reward tracer and replay buffer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


# run episodes
while env.T < 3000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_behavior(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= buffer.capacity:
            num_batches = int(4 * buffer.capacity / 32)  # 4 epochs per round
            for _ in range(num_batches):
                transition_batch = buffer.sample(32)
                Adv = simpletd.td_error(transition_batch)
                env.record_metrics(ppo_clip.update(transition_batch, Adv))
                env.record_metrics(simpletd.update(transition_batch))

            buffer.clear()

            # sync target networks
            pi_behavior.soft_update(pi, tau=0.1)
            v_targ.soft_update(v, tau=0.1)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 50000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(
            env=env, policy=pi.greedy, resize_to=(320, 420),
            filepath=gifs_filepath.format(T))
