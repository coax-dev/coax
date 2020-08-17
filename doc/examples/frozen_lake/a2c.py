import os

import coax
import jax
import jax.numpy as jnp
import gym
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


def func_v(S, is_training):
    value = hk.Sequential((hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    S = hk.one_hot(S, env.observation_space.n)
    return value(S)


def func_pi(S, is_training):
    logits = hk.Linear(env.action_space.n, w_init=jnp.zeros)
    S = hk.one_hot(S, env.observation_space.n)
    return {'logits': logits(S)}


# function approximators
pi = coax.Policy(func_pi, env.observation_space, env.action_space)
v = coax.V(func_v, env.observation_space)


# target network
v_targ = v.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)


# updaters
simple_td = coax.td_learning.SimpleTD(v, v_targ, optimizer=optix.adam(0.02))
vanillapg = coax.policy_objectives.VanillaPG(pi, optimizer=optix.adam(0.01))


# train
for ep in range(500):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if jnp.array_equal(s_next, s):
            r = -0.01

        # update
        tracer.add(s, a, r, done, logp)
        while tracer:
            transition_batch = tracer.pop()
            Adv = simple_td.td_error(transition_batch)
            vanillapg.update(transition_batch, Adv)
            simple_td.update(transition_batch)

            # sync target network
            v_targ.soft_update(v, tau=0.01)

        if done:
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(env.spec.max_episode_steps):

    # estimated state value
    print("  v(s) = {:.3f}".format(v(s)))

    # print individual action probabilities
    params = pi.dist_params(s)
    propensities = jax.nn.softmax(params['logits'])
    for i, p in enumerate(propensities):
        print("  π({:s}|s) = {:.3f}".format('LDRU'[i], p))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
