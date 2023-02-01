import os

# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet

import gymnasium
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from optax import adam


# the name of this script
name = 'dqn'

# env with preprocessing
env = gymnasium.make('PongNoFrameskip-v4', render_mode='rgb_array')  # AtariPreprocessing will do frame skipping
env = gymnasium.wrappers.AtariPreprocessing(env)
env = coax.wrappers.FrameStacking(env, num_frames=3)
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=108000 // 3)
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func(S, is_training):
    """ type-2 q-function: s -> q(s,.) """
    seq = hk.Sequential((
        coax.utils.diff_transform,
        hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
        hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
        hk.Flatten(),
        hk.Linear(256), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros),
    ))
    X = jnp.stack(S, axis=-1) / 255.  # stack frames
    return seq(X)


# function approximator
q = coax.Q(func, env)
pi = coax.EpsilonGreedy(q, epsilon=1.)

# target network
q_targ = q.copy()

# updater
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, optimizer=adam(3e-4))

# reward tracer and replay buffer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


# DQN exploration schedule (stepwise linear annealing)
epsilon = coax.utils.StepwiseLinearFunction((0, 1), (1000000, 0.1), (2000000, 0.01))


while env.T < 3000000:
    s, info = env.reset()
    pi.epsilon = epsilon(env.T)

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done or truncated)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) > 50000:  # buffer warm-up
            metrics = qlearning.update(buffer.sample(batch_size=32))
            env.record_metrics(metrics)

        if env.T % 10000 == 0:
            q_targ.soft_update(q, tau=1)

        if done or truncated:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 50000:
        T = env.T - env.T % 10000  # round to 10000s
        coax.utils.generate_gif(
            env=env, policy=pi, resize_to=(320, 420),
            filepath=f"./data/gifs/{name}/T{T:08d}.gif")
