# this is a script that I use for benchmarking my machine  -kris


import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix


# set some env vars
os.environ['JAX_PLATFORM_NAME'] = \
    os.environ.get('JAX_PLATFORM_NAME', 'gpu')        # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
coax.enable_logging()


# env with preprocessing
env = gym.make('PongNoFrameskip-v4')  # AtariPreprocessing does frame skipping
env = gym.wrappers.AtariPreprocessing(env, terminal_on_life_loss=True)
env = gym.wrappers.FrameStack(env, num_stack=3)
env = coax.wrappers.TrainMonitor(env)


class Func(coax.FuncApprox):
    def body(self, S):
        M = coax.utils.diff_transform_matrix(num_frames=3)
        seq = hk.Sequential([  # S.shape = [batch, num_stack, h, w]
            lambda x: jnp.dot(jnp.moveaxis(S / 255, 1, -1), M),  # [b, h, w, n]
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
q = coax.Q(func, qtype=2)
q_targ = q.copy()
pi = coax.EpsilonGreedy(q, epsilon=1.)

# updater
qlearning = coax.td_learning.QLearning(q, q_targ)

# replay buffer
buffer = coax.ExperienceReplayBuffer(env, capacity=50000, n=1, gamma=0.99)


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

        buffer.add(s, a, r, done)

        if len(buffer) > 1000:  # buffer warm-up
            qlearning.update(buffer.sample(batch_size=256))

        if env.period('target_model_sync', T_period=10000):
            q_targ.smooth_update(q, tau=1)

        if done:
            break

        s = s_next
