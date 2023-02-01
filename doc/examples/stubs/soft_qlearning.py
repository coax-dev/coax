import gymnasium
import coax
import haiku as hk
import jax
import jax.numpy as jnp
from optax import adam


# pick environment
env = gymnasium.make(...)
env = coax.wrappers.TrainMonitor(env)


def func_type1(S, A, is_training):
    # custom haiku function: s,a -> q(s,a)
    value = hk.Sequential([...])
    X = jax.vmap(jnp.kron)(S, A)  # or jnp.concatenate((S, A), axis=-1) or whatever you like
    return value(X)  # output shape: (batch_size,)


def func_type2(S, is_training):
    # custom haiku function: s -> q(s,.)
    value = hk.Sequential([...])
    return value(S)  # output shape: (batch_size, num_actions)


# function approximator
func = ...  # func_type1 or func_type2
q = coax.Q(func, env)
pi = coax.BoltzmannPolicy(q, temperature=0.1)


# specify how to update q-function
qlearning = coax.td_learning.SoftQLearning(q, optimizer=adam(0.001), temperature=pi.temperature)


# specify how to trace the transitions
cache = coax.reward_tracing.NStep(n=1, gamma=0.9)


for ep in range(100):
    pi.epsilon = ...  # exploration schedule
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # add transition to cache
        cache.add(s, a, r, done)

        # update
        while cache:
            transition_batch = cache.pop()
            metrics = qlearning.update(transition_batch)
            env.record_metrics(metrics)

        if done or truncated:
            break

        s = s_next
