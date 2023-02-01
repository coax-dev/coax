"""

This script is a JAX port of the original Tensorflow-based script:

    https://gist.github.com/heerad/1983d50c6657a55298b67e69a2ceeb44#file-ddpg-pendulum-v0-py


"""

import os
import json
import random
from collections import deque
from functools import partial
from copy import deepcopy

import gymnasium
import numpy as onp
import jax
import jax.numpy as jnp
import haiku as hk
import coax
import optax


class hparams:
    gamma = 0.99                # reward discount factor
    h1_actor = 8                # hidden layer 1 size for the actor
    h2_actor = 8                # hidden layer 2 size for the actor
    h3_actor = 8                # hidden layer 3 size for the actor
    h1_critic = 8               # hidden layer 1 size for the critic
    h2_critic = 8               # hidden layer 2 size for the critic
    h3_critic = 8               # hidden layer 3 size for the critic
    lr_actor = 1e-3             # learning rate for the actor
    lr_critic = 1e-3            # learning rate for the critic
    lr_decay = 1                # learning rate decay (per episode)
    l2_reg_actor = 1e-6         # L2 regularization factor for the actor
    l2_reg_critic = 1e-6        # L2 regularization factor for the critic
    dropout_actor = 0           # dropout rate for actor (0 = no dropout)
    dropout_critic = 0          # dropout rate for critic (0 = no dropout)
    num_episodes = 15000        # number of episodes
    max_steps_ep = 10000    # default max number of steps per episode (unless env has a lower hardcoded limit)
    tau = 1e-2              # soft target update rate
    train_every = 1         # number of steps to run the policy (and collect experience) before updating network weights
    replay_memory_capacity = int(1e5)   # capacity of experience replay memory
    minibatch_size = 1024   # size of minibatch from experience replay memory for updates
    noise_decay = 0.9999    # decay rate (per step) of the scale of the exploration noise process
    exploration_mu = 0.0    # mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
    exploration_theta = 0.15 # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
    exploration_sigma = 0.2  # sigma parameter for the exploration noise process: dXt = theta*(mu-Xt )*dt + sigma*dWt

    @classmethod
    def to_json(cls, dirpath):
        dct = {k: v for k, v in cls.__dict__.items() if not k.startswith('_') and k != 'to_json'}
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open(os.path.join(dirpath, 'hparams.pkl'), 'w') as f:
            json.dump(dct, f)


# filepaths etc
experiment_id, _ = os.path.splitext(__file__)
tensorboard_dir = f"./data/tensorboard/{experiment_id}"
coax.utils.enable_logging(experiment_id)


# the MDP
env = gymnasium.make('Pendulum-v1', render_mode='rgb_array')
env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


# store hparams to disk, just in case it gets lost
hparams.to_json(env.tensorboard.logdir)


# set seeds to 0
env.seed(0)
onp.random.seed(0)
# onp.set_printoptions(threshold=sys.maxsize)


# used for O(1) popleft() operation
replay_memory = deque(maxlen=hparams.replay_memory_capacity)


def add_to_memory(s, a, r, done, s_next):
    replay_memory.append((s, a, r, float(done), s_next))


def sample_from_memory(minibatch_size):
    transitions = random.sample(replay_memory, minibatch_size)
    # S, A, R, D, S_next = (jnp.stack(x, axis=0) for x in zip(*transitions))
    S = onp.stack([t[0] for t in transitions], axis=0)
    A = onp.stack([t[1] for t in transitions], axis=0)
    R = onp.stack([t[2] for t in transitions], axis=0)
    D = onp.stack([t[3] for t in transitions], axis=0)
    S_next = onp.stack([t[4] for t in transitions], axis=0)
    In = hparams.gamma * (1 - D)
    return coax.reward_tracing.TransitionBatch(S, A, None, R, In, S_next, None, None)


####################################################################################################
# forward-passes

def pi(S, is_training):
    rng1, rng2, rng3 = hk.next_rng_keys(3)
    shape = env.action_space.shape
    rate = hparams.dropout_actor * is_training
    seq = hk.Sequential((
        hk.Linear(hparams.h1_actor), jax.nn.relu, partial(hk.dropout, rng1, rate),
        hk.Linear(hparams.h2_actor), jax.nn.relu, partial(hk.dropout, rng2, rate),
        hk.Linear(hparams.h3_actor), jax.nn.relu, partial(hk.dropout, rng3, rate),
        hk.Linear(onp.prod(shape)), hk.Reshape(shape),
        # lambda x: low + (high - low) * jax.nn.sigmoid(x),  # disable: BoxActionsToReals
    ))
    return seq(S)  # batch of actions


def q(S, A, is_training):
    rng1, rng2, rng3 = hk.next_rng_keys(3)
    rate = hparams.dropout_critic * is_training
    seq = hk.Sequential((
        hk.Linear(hparams.h1_critic), jax.nn.relu, partial(hk.dropout, rng1, rate),
        hk.Linear(hparams.h2_critic), jax.nn.relu, partial(hk.dropout, rng2, rate),
        hk.Linear(hparams.h3_critic), jax.nn.relu, partial(hk.dropout, rng3, rate),
        hk.Linear(1), jnp.ravel,
    ))
    flatten = hk.Flatten()
    X_sa = jnp.concatenate([flatten(S), jnp.tanh(flatten(A))], axis=1)
    return seq(X_sa)


# dummy input (with batch axis) to initialize params
rngs = hk.PRNGSequence(13)
S = jnp.zeros((1,) + env.observation_space.shape)
A = jnp.zeros((1,) + env.action_space.shape)


# Haiku-transform actor
pi = hk.transform(pi, apply_rng=True)
params_pi = pi.init(next(rngs), S, True)
pi = jax.jit(pi.apply, static_argnums=3)


# Haiku-transform critic
q = hk.transform(q, apply_rng=True)
params_q = q.init(next(rngs), S, A, True)
q = jax.jit(q.apply, static_argnums=4)


# target-network params
target_params_pi = deepcopy(params_pi)
target_params_q = deepcopy(params_q)


@jax.jit
def soft_update(target_params, primary_params, tau=1.0):
    return jax.tree_map(lambda a, b: a + tau * (b - a), target_params, primary_params)


####################################################################################################
# loss functions and optimizers

def loss_q(params_q, target_params_q, target_params_pi, rng, transition_batch):
    rngs = hk.PRNGSequence(rng)
    S, A, _, Rn, In, S_next, _, _ = transition_batch
    A_next = pi(target_params_pi, next(rngs), S_next, False)
    Q_next = q(target_params_q, next(rngs), S_next, A_next, False)
    target = Rn + In * Q_next
    pred = q(params_q, next(rngs), S, A, True)
    return jnp.mean(jnp.square(pred - target))


def loss_pi(params_pi, target_params_q, rng, transition_batch):
    rngs = hk.PRNGSequence(rng)
    S, A = transition_batch[:2]
    A = pi(params_pi, next(rngs), S, True)
    Q = q(target_params_q, next(rngs), S, A, False)
    return -jnp.mean(Q)  # flip sign


@partial(jax.jit, static_argnums=2)
def update_fn(params, grads, optimizer, optimizer_state):
    updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_optimizer_state


# loss + gradient functions
loss_and_grad_pi = jax.jit(jax.value_and_grad(loss_pi))
loss_and_grad_q = jax.jit(jax.value_and_grad(loss_q))


# actor optimizer
optimizer_pi = optax.adam(hparams.lr_actor)
optimizer_state_pi = optimizer_pi.init(params_pi)


# critic optimizer
optimizer_q = optax.adam(hparams.lr_critic)
optimizer_state_q = optimizer_q.init(params_q)


def update(transition_batch):
    """ high-level utility function for updating the actor-critic """
    global params_pi, optimizer_state_pi, params_q, optimizer_state_q

    # update the actor
    loss_pi, grads_pi = loss_and_grad_pi(params_pi, target_params_q, next(rngs), transition_batch)
    params_pi, optimizer_state_pi = update_fn(params_pi, grads_pi, optimizer_pi, optimizer_state_pi)

    # update the critic
    loss_q, grads_q = loss_and_grad_q(
        params_q, target_params_q, target_params_pi, next(rngs), transition_batch)
    params_q, optimizer_state_q = update_fn(params_q, grads_q, optimizer_q, optimizer_state_q)

    return {'pi/loss': loss_pi, 'q/loss': loss_q}


####################################################################################################
# action noise

def noise(a, ep):
    a = jnp.asarray(a)
    shape = env.action_space.shape
    mu, sigma, theta = hparams.exploration_mu, hparams.exploration_sigma, hparams.exploration_theta
    scale = hk.get_state('scale', shape=(), dtype=a.dtype, init=jnp.ones)
    noise = hk.get_state('noise', shape=a.shape, dtype=a.dtype, init=jnp.zeros)
    scale = scale * hparams.noise_decay
    noise = theta * (mu - noise) + sigma * jax.random.normal(hk.next_rng_key(), shape)
    hk.set_state('scale', scale)
    hk.set_state('noise', noise)
    return a + noise * scale


# Haiku-transform
noise = hk.transform_with_state(noise)
noise_params, noise_init_state = noise.init(next(rngs), env.action_space.sample(), 13)
noise = jax.jit(partial(noise.apply, noise_params))  # params are trivial, so we'll discard them
del noise_params


def sample_action(s, with_noise=False):
    global noise_state
    S = jnp.expand_dims(s, axis=0)
    A = pi(params_pi, next(rngs), S, False)
    a = jnp.squeeze(A, axis=0)
    if not with_noise:
        return a
    a, noise_state = noise(noise_state, next(rngs), a, env.ep)
    return onp.clip(a, env.action_space.low, env.action_space.high)  # ordinary numpy array


####################################################################################################
# train


for _ in range(hparams.num_episodes):
    noise_state = noise_init_state  # reset noise state
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = sample_action(s, with_noise=True)
        s_next, r, done, truncated, info = env.step(a)

        # add to replay buffer
        add_to_memory(s, a, r, done, s_next)

        if len(replay_memory) >= max(hparams.minibatch_size, 5000):
            transition_batch = sample_from_memory(hparams.minibatch_size)
            metrics = update(transition_batch)
            env.record_metrics(metrics)

            # update target networks
            target_params_pi = soft_update(target_params_pi, params_pi, hparams.tau)
            target_params_q = soft_update(target_params_q, params_q, hparams.tau)

        if done or truncated:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(
            env=env, policy=sample_action,
            filepath=os.path.join(env.tensorboard.logdir, f'T{T:08d}.gif'))
