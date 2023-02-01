import gymnasium
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from numpy import prod
import optax


# the name of this script
name = 'dsac'

# the Pendulum MDP
env = gymnasium.make('Pendulum-v1', render_mode='rgb_array')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")

quantile_embedding_dim = 64
layer_size = 256
num_quantiles = 32


def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape) * 2, w_init=jnp.zeros),
        hk.Reshape((*env.action_space.shape, 2)),
    ))
    x = seq(S)
    mu, logvar = x[..., 0], x[..., 1]
    return {'mu': mu, 'logvar': logvar}


def quantile_net(x, quantile_fractions):
    quantiles_emb = coax.utils.quantile_cos_embedding(
        quantile_fractions, quantile_embedding_dim)
    quantiles_emb = hk.Linear(x.shape[-1])(quantiles_emb)
    quantiles_emb = jax.nn.relu(quantiles_emb)
    x = x[:, None, :] * quantiles_emb
    x = hk.Linear(layer_size)(x)
    x = jax.nn.relu(x)
    return x


def func_q(S, A, is_training):
    encoder = hk.Sequential((
        hk.Flatten(),
        hk.Linear(layer_size),
        jax.nn.relu
    ))
    quantile_fractions = coax.utils.quantiles_uniform(rng=hk.next_rng_key(),
                                                      batch_size=S.shape[0],
                                                      num_quantiles=num_quantiles)
    X = jnp.concatenate((S, A), axis=-1)
    x = encoder(X)
    quantile_x = quantile_net(x, quantile_fractions=quantile_fractions)
    quantile_values = hk.Linear(1)(quantile_x)
    return {'values': quantile_values.squeeze(axis=-1),
            'quantile_fractions': quantile_fractions}


# main function approximators
pi = coax.Policy(func_pi, env)
q1 = coax.StochasticQ(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate,
                      value_range=None, num_bins=num_quantiles)
q2 = coax.StochasticQ(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate,
                      value_range=None, num_bins=num_quantiles)

# target network
q1_targ = q1.copy()
q2_targ = q2.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9, record_extra_info=True)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=50000)
alpha = 0.2
policy_regularizer = coax.regularizers.NStepEntropyRegularizer(pi,
                                                               beta=alpha / tracer.n,
                                                               gamma=tracer.gamma,
                                                               n=[tracer.n])

# updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
qlearning1 = coax.td_learning.SoftClippedDoubleQLearning(
    q1, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
    loss_function=coax.value_losses.mse, optimizer=optax.adam(3e-4),
    policy_regularizer=policy_regularizer)
qlearning2 = coax.td_learning.SoftClippedDoubleQLearning(
    q2, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
    loss_function=coax.value_losses.mse, optimizer=optax.adam(3e-4),
    policy_regularizer=policy_regularizer)
soft_pg = coax.policy_objectives.SoftPG(pi, [q1_targ, q2_targ], optimizer=optax.adam(
    1e-3), regularizer=coax.regularizers.NStepEntropyRegularizer(pi,
                                                                 beta=alpha / tracer.n,
                                                                 gamma=tracer.gamma,
                                                                 n=jnp.arange(tracer.n)))


# train
while env.T < 1000000:
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 5000:
            transition_batch = buffer.sample(batch_size=256)

            # init metrics dict
            metrics = {}

            # flip a coin to decide which of the q-functions to update
            qlearning = qlearning1 if jax.random.bernoulli(q1.rng) else qlearning2
            metrics.update(qlearning.update(transition_batch))

            # delayed policy updates
            if env.T >= 7500 and env.T % 4 == 0:
                metrics.update(soft_pg.update(transition_batch))

            env.record_metrics(metrics)

            # sync target networks
            q1_targ.soft_update(q1, tau=0.005)
            q2_targ.soft_update(q2, tau=0.005)

        if done or truncated:
            break

        s = s_next

    generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000  # round to 10000s
        coax.utils.generate_gif(
            env=env, policy=pi, filepath=f"./data/gifs/{name}/T{T:08d}.gif")
