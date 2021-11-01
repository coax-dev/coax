import coax
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
from optax import adam


# the name of this script
name = 'iqn'
# the cart-pole MDP
env = gym.make('CartPole-v0')
env = coax.wrappers.TrainMonitor(
    env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")
quantile_embedding_dim = 32
layer_size = 256
num_quantiles = 32


def quantile_net(x, quantiles):
    x_size = x.shape[-1]
    x_tiled = jnp.tile(x[:, None, :], [num_quantiles, 1])
    quantile_net = jnp.tile(quantiles[..., None], [1, quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, quantile_embedding_dim + 1, 1)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = hk.Linear(x_size)(quantile_net)
    quantile_net = hk.LayerNorm(axis=-1, create_scale=True,
                                create_offset=True)(quantile_net)
    quantile_net = jax.nn.sigmoid(quantile_net)
    x = x_tiled * quantile_net
    x = hk.Linear(x_size)(x)
    x = jax.nn.relu(x)
    return x


def func(S, A, is_training):
    """ type-1 q-function: (s,a) -> q(s,a) """
    encoder = hk.Sequential((
        hk.Flatten(), hk.Linear(layer_size), jax.nn.relu
    ))
    quantile_fractions = coax.utils.quantile_func_iqn(S=S, rng=hk.next_rng_key(),
                                                      is_training=is_training,
                                                      num_quantiles=num_quantiles)
    X = jax.vmap(jnp.kron)(S, A)
    x = encoder(X)
    quantile_x = quantile_net(x, quantiles=quantile_fractions)
    quantile_values = hk.Linear(1, w_init=jnp.zeros)(quantile_x)
    return {'values': quantile_values.squeeze(axis=-1),
            'quantile_fractions': quantile_fractions}


# quantile value function and its derived policy
q = coax.StochasticQ(func, env, num_bins=num_quantiles, value_range=None)
pi = coax.BoltzmannPolicy(q)

# target network
q_targ = q.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

# updater
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, optimizer=adam(0.001))


# train
for ep in range(1000):
    s = env.reset()
    # pi.epsilon = max(0.01, pi.epsilon * 0.95)
    # env.record_metrics({'EpsilonGreedy/epsilon': pi.epsilon})

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # extend last reward as asymptotic best-case return
        if t == env.spec.max_episode_steps - 1:
            assert done
            r = 1 / (1 - tracer.gamma)  # gamma + gamma^2 + gamma^3 + ... = 1 / (1 - gamma)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 100:
            transition_batch = buffer.sample(batch_size=32)
            metrics = qlearning.update(transition_batch)
            env.record_metrics(metrics)

        # sync target network
        q_targ.soft_update(q, tau=0.01)

        if done:
            break

        s = s_next

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        break


# run env one more time to render
coax.utils.generate_gif(env, policy=pi, filepath=f"./data/{name}.gif", duration=25)
