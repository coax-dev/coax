import os

import gym
import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
from jax.experimental import optix


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/ppo"
gifs_filepath = "./data/gifs/ppo/T{:08d}.gif"


# env with preprocessing
env = gym.make('Pendulum-v0')  # AtariPreprocessing does frame skipping
# env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


def func_pi(S, is_training):
    shared = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
    ))
    mu = hk.Sequential((
        shared,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    logvar = hk.Sequential((
        shared,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    return {'mu': mu(S), 'logvar': logvar(S)}


def func_v(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return seq(S)


# define function approximators
pi = coax.Policy(func_pi, env.observation_space, env.action_space)
v = coax.V(func_v, env.observation_space)


# target network
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=512)


# policy regularizer (avoid premature exploitation)
policy_reg = coax.policy_regularizers.EntropyRegularizer(pi, beta=0.01)


# updaters
simpletd = coax.td_learning.SimpleTD(v, optimizer=optix.adam(1e-3))
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=policy_reg, optimizer=optix.adam(1e-4))


# train
while env.T < 1000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_targ(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # trace rewards
        tracer.add(s, a, r, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= buffer.capacity:
            for _ in range(int(4 * buffer.capacity / 32)):  # 4 passes per round
                transition_batch = buffer.sample(batch_size=32)
                Adv = simpletd.td_error(transition_batch)

                metrics = {}
                metrics.update(ppo_clip.update(transition_batch, Adv))
                metrics.update(simpletd.update(transition_batch))
                env.record_metrics(metrics)

            buffer.clear()
            pi_targ.soft_update(pi, tau=0.1)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(env=env, policy=pi.greedy, filepath=gifs_filepath.format(T))
