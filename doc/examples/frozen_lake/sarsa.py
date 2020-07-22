import os
import jax.numpy as jnp
import gym
from gym.envs.toy_text.frozen_lake import UP, DOWN, LEFT, RIGHT

import coax

# set some env vars
os.environ['JAX_PLATFORM_NAME'] = 'cpu'   # tell JAX to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


# the MDP
actions = {LEFT: 'L', RIGHT: 'R', UP: 'U', DOWN: 'D'}
env = gym.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)

# show logs from TrainMonitor
coax.enable_logging()


# define function approximators
func = coax.FuncApprox(env, learning_rate=0.2)
q = coax.Q(func)
pi = coax.EpsilonGreedy(q, epsilon=0.1)
# pi = coax.BoltzmannPolicy(q)


# experience tracer
cache = coax.reward_tracing.NStepCache(env, n=1, gamma=0.9)


# updater
sarsa = coax.td_learning.Sarsa(q)


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

        cache.add(s, a, r, done)
        while cache:
            transition_batch = cache.pop()
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
        print("  q(s,{:s}) = {:.3f}".format(actions[i], q_))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
