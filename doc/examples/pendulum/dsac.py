import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from numpy import prod
import optax

# the name of this script
name = 'dsac'

# the Pendulum MDP
env = gym.make('Pendulum-v1')
env = coax.wrappers.TrainMonitor(
    env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")
num_bins = 51


def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(2 * prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape((2, *env.action_space.shape)),
    ))
    x = seq(S)
    mu, logvar = x[:, 0], x[:, 1]
    return {'mu': mu, 'logvar': logvar}


def func_q(S, A, is_training):
    logits = hk.Sequential((hk.Linear(8), jax.nn.relu,
                            hk.Flatten(), hk.Linear(num_bins, w_init=jnp.zeros)))
    X = jax.vmap(jnp.kron)(S, A)  # S and A are one-hot encoded
    return {'logits': logits(X)}


# main function approximators
pi = coax.Policy(func_pi, env)
q1 = coax.StochasticQ(
    func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate,
    value_range=(-10, 0), num_bins=num_bins)
q2 = coax.StochasticQ(
    func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate,
    value_range=(-10, 0), num_bins=num_bins)


# target network
q1_targ = q1.copy()
q2_targ = q2.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9, record_extra_info=True)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=25000)


# updaters (use current pi to update the q-functions and use sampled action in contrast to TD3)
qlearning1 = coax.td_learning.ClippedDoubleQLearning(
    q1, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
    loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3))
qlearning2 = coax.td_learning.ClippedDoubleQLearning(
    q2, pi_targ_list=[pi], q_targ_list=[q1_targ, q2_targ],
    loss_function=coax.value_losses.mse, optimizer=optax.adam(1e-3))
soft_pg = coax.policy_objectives.DeterministicPG(pi, q1_targ, optimizer=optax.adam(
    1e-3), regularizer=coax.regularizers.NStepEntropyRegularizer(pi, beta=0.2 / tracer.n,
                                                                 gamma=tracer.gamma, n=tracer.n))


# train
while env.T < 1000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 5000:
            transition_batch = buffer.sample(batch_size=128)

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
            q1_targ.soft_update(q1, tau=0.001)
            q2_targ.soft_update(q2, tau=0.001)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000  # round to 10000s
        coax.utils.generate_gif(
            env=env, policy=pi, filepath=f"./data/gifs/{name}/T{T:08d}.gif")
