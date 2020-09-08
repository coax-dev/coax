import os

import coax
import gym
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from coax.value_losses import mse


# set some env vars
os.environ['JAX_PLATFORM_NAME'] = 'cpu'   # tell JAX to use CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


# the cart-pole MDP
env = gym.make('CartPole-v0')
env = coax.wrappers.TrainMonitor(env, log_all_metrics=True)


def func_v(S, is_training):
    kinetic = S[:, 3] ** 2        # angular velocity squared
    potential = hk.Sequential((
        lambda x: x[:, :3] ** 2,  # drop angular velocity
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return -kinetic + potential(S)


def func_p(S, A, is_training):
    linear = hk.Linear(4, w_init=jnp.zeros)
    mu = S + linear(A)
    return {'mu': mu, 'logvar': jnp.full_like(mu, -jnp.inf)}  # deterministic (var=0)


def func_r(S, A, is_training):
    mu = jnp.ones(S.shape[0])  # CartPole yields r=1 at every time step (no need to learn)
    return {'mu': mu, 'logvar': jnp.full_like(mu, -jnp.inf)}  # deterministic (var=0)


# function approximators
v = coax.V(func_v, env.observation_space)
p = coax.DynamicsModel(func_p, env.observation_space, env.action_space)
r = coax.RewardModel(func_r, env.observation_space, env.action_space, env.reward_range)


# composite objects
q = coax.SuccessorStateQ(v, p, r, gamma=0.9)
pi = coax.EpsilonGreedy(q, epsilon=0.1)


# reward tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=q.gamma)


# updaters
adam = optax.chain(optax.apply_every(k=32), optax.adam(0.001))
simple_td = coax.td_learning.SimpleTD(v, loss_function=mse, optimizer=adam)

sgd = optax.sgd(0.01, momentum=0.9, nesterov=True)
model_updater = coax.model_updaters.SimpleModelUpdater(p, optimizer=sgd)


while env.T < 100000:
    s = env.reset()
    env.render()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)
        env.render()
        env.record_metrics({'p/loss': jnp.linalg.norm(s_next - p(s, a))})

        if t == env.spec.max_episode_steps - 1:
            r = 1. / (1. - tracer.gamma)  # best case asymptotic return

        tracer.add(s, a, r, done)
        while tracer:
            transition_batch = tracer.pop()
            env.record_metrics(simple_td.update(transition_batch))
            env.record_metrics(model_updater.update(transition_batch))

        if done:
            env.logger.info(f"p.params: {p.params}")
            env.logger.info(f"v.params: {v.params}")
            break

        s = s_next


# run env one more time to render
coax.utils.generate_gif(env, pi.mode, filepath="data/model_based.gif", duration=25)
