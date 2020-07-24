import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix
from ray.rllib.env.atari_wrappers import wrap_deepmind


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/dqn_type1"
gifs_filepath = "./data/gifs/dqn_type1/T{:08d}.gif"
coax.enable_logging('dqn_type1')


# env with preprocessing
env = gym.make('PongNoFrameskip-v4')  # wrap_deepmind will do frame skipping
env = wrap_deepmind(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir=tensorboard_dir)


class Func(coax.FuncApprox):
    def body(self, S, is_training):
        M = coax.utils.diff_transform_matrix(num_frames=S.shape[-1])
        seq = hk.Sequential([  # S.shape = [batch, h, w, num_stack]
            lambda x: jnp.dot(S / 255, M),  # [b, h, w, n]
            hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
            hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256), jax.nn.relu,
        ])
        return seq(S)

    def optimizer(self):
        return optix.adam(learning_rate=0.00025)


# function approximators
func = Func(env)
q = coax.Q(func, qtype=1)  # <--- different from standard DQN
q_targ = q.copy()
pi = coax.EpsilonGreedy(q, epsilon=1.)

# updater
qlearning = coax.td_learning.QLearning(q, q_targ)

# reward tracer and replay buffer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


# DQN exploration schedule (stepwise linear annealing)
def epsilon(T):
    M = 1000000
    if T < M:
        return 1 - 0.9 * T / M
    if T < 2 * M:
        return 0.1 - 0.09 * (T - M) / M
    return 0.01


while env.T < 3000000:
    s = env.reset()
    pi.epsilon = epsilon(env.T)

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) > 50000:  # buffer warm-up
            metrics = qlearning.update(buffer.sample(batch_size=32))
            env.record_metrics(metrics)

        if env.period('target_model_sync', T_period=10000):
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
