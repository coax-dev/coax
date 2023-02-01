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
pi = coax.EpsilonGreedy(q, epsilon=1.0)  # epsilon will be updated


# target network
q_targ = q.copy()


# specify how to update q-function
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, optimizer=optax.adam(0.001))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.PrioritizedReplayBuffer(capacity=1000000, alpha=0.6, beta=0.4)


# schedules for pi.epsilon and buffer.beta
epsilon = coax.utils.StepwiseLinearFunction((0, 1), (1000000, 0.1), (2000000, 0.01))
beta = coax.utils.StepwiseLinearFunction((0, 0.4), (1000000, 1))


while env.T < 3000000:
    pi.epsilon = epsilon(env.T)
    buffer.beta = beta(env.T)

    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done)
        while tracer:
            transition = tracer.pop()
            buffer.add(transition, qlearning.td_error(transition))

        # update
        transition_batch = buffer.sample(batch_size=32)
        metrics, td_error = qlearning.update(transition_batch, return_td_error=True)
        buffer.update(transition_batch.idx, td_error)
        env.record_metrics(metrics)

        # periodically sync target model
        if env.ep % 10 == 0:
            q_targ.soft_update(q, tau=1.0)

        if done or truncated:
            break

        s = s_next
