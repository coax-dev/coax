import gymnasium
import coax
import optax
import haiku as hk
import jax
import jax.numpy as jnp


# pick environment
env = gymnasium.make(...)
env = coax.wrappers.TrainMonitor(env)

# choose iqn hyperparameters
quantile_embedding_dim = 32
num_quantiles = 32


def func_type1(S, A, is_training):
    # custom haiku function: s,a -> q(s,a)
    net = hk.Sequential([...])
    X = jax.vmap(jnp.kron)(S, A)  # or jnp.concatenate((S, A), axis=-1) or whatever you like
    quantile_values, quantile_fractions = net(X)
    return {'values': quantile_values,  # output shape: (batch_size, num_quantiles)
            'quantile_fractions': quantile_fractions}


def func_type2(S, is_training):
    # custom haiku function: s -> q(s,.)
    quantile_values, quantile_fractions = hk.Sequential([...])
    return {'values': quantile_values,  # output shape: (batch_size, num_actions, num_quantiles)
            'quantile_fractions': quantile_fractions}


# function approximator
func = ...  # func_type1 or func_type2

# quantile value function and its derived policy
q = coax.StochasticQ(func, env, num_bins=num_quantiles, value_range=None)
pi = coax.BoltzmannPolicy(q)

# target network
q_targ = q.copy()

# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

# specify how to update q-function
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, optimizer=optax.adam(0.001))


for ep in range(1000):
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

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

        if done or truncated:
            break

        s = s_next

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        break
