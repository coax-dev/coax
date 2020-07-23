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
pi = coax.BoltzmannPolicy(q, tau=0.1)


# experience tracer
tracer = coax.reward_tracing.NStepCache(n=1, gamma=0.9)


# updater
esarsa = coax.td_learning.ExpectedSarsa(q, pi)


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
            esarsa.update(transition_batch)

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
