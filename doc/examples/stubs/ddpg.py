import gymnasium
import coax
import optax
import haiku as hk
import jax.numpy as jnp


# pick environment
env = gymnasium.make(...)
env = coax.wrappers.TrainMonitor(env)


def func_pi(S, is_training):
    # custom haiku function (for continuous actions in this example)
    mu = hk.Sequential([...])(S)  # mu.shape: (batch_size, *action_space.shape)
    return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}  # deterministic policy


def func_q(S, A, is_training):
    # custom haiku function
    value = hk.Sequential([...])
    return value(S)  # output shape: (batch_size,)


# define function approximator
pi = coax.Policy(func_pi, env)
q = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)


# target networks
pi_targ = pi.copy()
q_targ = q.copy()


# specify how to update policy and value function
determ_pg = coax.policy_objectives.DeterministicPG(pi, q, optimizer=optax.adam(0.001))
qlearning = coax.td_learning.QLearning(q, pi_targ, q_targ, optimizer=optax.adam(0.002))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=1000000)


# action noise
noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=0.2, theta=0.15)


for ep in range(100):
    s, info = env.reset()
    noise.reset()
    noise.sigma *= 0.99  # slowly decrease noise scale

    for t in range(env.spec.max_episode_steps):
        a = noise(pi(s))
        s_next, r, done, truncated, info = env.step(a)

        # add transition to buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # update
        transition_batch = buffer.sample(batch_size=32)
        metrics_q = qlearning.update(transition_batch)
        metrics_pi = determ_pg.update(transition_batch)
        env.record_metrics(metrics_q)
        env.record_metrics(metrics_pi)

        # periodically sync target models
        if ep % 10 == 0:
            pi_targ.soft_update(pi, tau=1.0)
            q_targ.soft_update(q, tau=1.0)

        if done or truncated:
            break

        s = s_next
