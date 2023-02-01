import gymnasium
import coax
import optax
import haiku as hk
import jax
import jax.numpy as jnp


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
pi = coax.EpsilonGreedy(q, epsilon=0.1)


# specify how to update q-function
sarsa = coax.td_learning.Sarsa(q, optimizer=optax.adam(0.001))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)


for ep in range(100):
    pi.epsilon = ...  # exploration schedule
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards to create training data
        tracer.add(s, a, r, done)

        # update
        while tracer:
            transition_batch = tracer.pop()
            metrics = sarsa.update(transition_batch)
            env.record_metrics(metrics)

        if done or truncated:
            break

        s = s_next
