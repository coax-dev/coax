import os

import coax
import gym
import jax.numpy as jnp
import haiku as hk
from jax.experimental import optix


# set some env vars
os.environ['JAX_PLATFORM_NAME'] = 'cpu'   # tell JAX to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


# the MDP
env = gym.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)

# show logs from TrainMonitor
coax.enable_logging()


def func(S, A, is_training):
    value = hk.Sequential((hk.Flatten(), hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    S = hk.one_hot(S, env.observation_space.n)
    X = jnp.kron(S, A)  # A is already one-hot encoded
    return value(X)


# function approximator
q = coax.Q(func, env.observation_space, env.action_space)
pi = coax.EpsilonGreedy(q, epsilon=0.1)


# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)


# updater
sarsa = coax.td_learning.Sarsa(q, optimizer=optix.adam(0.02))


# train
for ep in range(500):
    pi.epsilon = max(0.1, 1 - env.ep / 400)
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if jnp.array_equal(s_next, s):
            r = -0.01

        # update
        tracer.add(s, a, r, done)
        while tracer:
            transition_batch = tracer.pop()
            sarsa.update(transition_batch)

        if done:
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(env.spec.max_episode_steps):

    # print individual state-action values
    for i, q_ in enumerate(q(s)):
        print("  q(s,{:s}) = {:.3f}".format('LDRU'[i], q_))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
