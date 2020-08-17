import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from numpy import prod
from jax.experimental import optix


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/ddpg"
gifs_filepath = "./data/gifs/ddpg/T{:08d}.gif"


# env with preprocessing
env = gym.make('Pendulum-v0')  # AtariPreprocessing does frame skipping
# env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    mu = seq(S)
    return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}  # (almost) deterministic


def func_q(S, A, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    X = jnp.concatenate((S, A), axis=-1)
    return seq(X)


# main function approximators
pi = coax.Policy(func_pi, env.observation_space, env.action_space)
q = coax.Q(func_q, env.observation_space, env.action_space)


# target network
q_targ = q.copy()
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=25000)


# updaters
qlearning = coax.td_learning.QLearningMode(q, pi_targ, q_targ,
                                           loss_function=coax.value_losses.mse,
                                           optimizer=optix.adam(1e-3))
determ_pg = coax.policy_objectives.DeterministicPG(pi, q_targ, optimizer=optix.adam(1e-4))


# action noise
noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=0.2, theta=0.15)


# train
while env.T < 1000000:
    s = env.reset()
    noise.reset()
    noise.sigma *= 0.99  # slowly decrease noise scale

    for t in range(env.spec.max_episode_steps):
        a = noise(pi(s))
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 5000:
            transition_batch = buffer.sample(batch_size=1024)

            metrics = {'OrnsteinUhlenbeckNoise/sigma': noise.sigma}
            metrics.update(determ_pg.update(transition_batch))
            metrics.update(qlearning.update(transition_batch))
            env.record_metrics(metrics)

            # sync target networks
            q_targ.soft_update(q, tau=0.001)
            pi_targ.soft_update(pi, tau=0.001)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(env=env, policy=pi.greedy, filepath=gifs_filepath.format(T))
