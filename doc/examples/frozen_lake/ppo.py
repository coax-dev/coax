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
v = coax.V(func)


# create copies
pi_old = pi.copy()  # behavior policy
v_targ = v.copy()   # target network


# experience tracer
tracer = coax.reward_tracing.NStepCache(n=1, gamma=0.9)


# updaters
value_td = coax.td_learning.ValueTD(v, v_targ)
ppo_clip = coax.policy_objectives.PPOClip(pi)


# train
for ep in range(250):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_old(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if jnp.array_equal(s_next, s):
            r = -0.01

        # update
        tracer.add(s, a, r, done, logp)
        while tracer:
            transition_batch = tracer.pop()
            Adv = value_td.td_error(transition_batch)
            ppo_clip.update(transition_batch, Adv)
            value_td.update(transition_batch)

        # sync copies
        if env.T % 20 == 0:
            v_targ.soft_update(v, tau=0.5)
            pi_old.soft_update(pi, tau=0.5)

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
        print("  π({:s}|s) = {:.3f}".format(actions[i], p))

    a = pi.greedy(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
