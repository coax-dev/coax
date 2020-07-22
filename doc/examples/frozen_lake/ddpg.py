import os
import jax
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
func = coax.FuncApprox(env, learning_rate=0.5)
pi = coax.Policy(func)
q = coax.Q(func)


# target network
q_targ = q.copy()
pi_targ = pi.copy()


# experience tracer
cache = coax.reward_tracing.NStepCache(n=1, gamma=0.9)


# updaters
sarsa = coax.td_learning.Sarsa(q, q_targ)
ddpg = coax.policy_objectives.DeterministicPG(pi, q)


# train
for ep in range(250):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if jnp.array_equal(s_next, s):
            r = -0.01

        # update
        cache.add(s, a, r, done)
        while cache:
            transition_batch = cache.pop()
            ddpg.update(transition_batch)
            sarsa.update(transition_batch)

        # sync copies
        if env.T % 20 == 0:
            q_targ.soft_update(q, tau=0.5)
            pi_targ.soft_update(pi, tau=0.5)

        if done:
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(env.spec.max_episode_steps):

    # print individual action probabilities
    params = pi.dist_params(s)
    propensities = jax.nn.softmax(params['logits'])
    for i, p in enumerate(propensities):
        print("  π({:s}|s) = {:.3f}".format(actions[i], p))
    for i, p in enumerate(q(s)):
        print("  q(s,{:s}) = {:.3f}".format(actions[i], p))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
