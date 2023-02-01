import gymnasium
import coax
import jax
import jax.numpy as jnp
import haiku as hk
import optax


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
q1 = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)
q2 = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)


# target networks
pi_targ = pi.copy()
q1_targ = q1.copy()
q2_targ = q2.copy()


# specify how to update policy and value function
determ_pg = coax.policy_objectives.DeterministicPG(pi, q1, optimizer=optax.adam(0.001))
qlearning1 = coax.td_learning.ClippedDoubleQLearning(
    q1, q_targ_list=[q1_targ, q2_targ], optimizer=optax.adam(0.001))
qlearning2 = coax.td_learning.ClippedDoubleQLearning(
    q2, q_targ_list=[q1_targ, q2_targ], optimizer=optax.adam(0.001))


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
        if len(buffer) >= 128:
            transition_batch = buffer.sample(batch_size=32)

            # flip a coin to decide which of the q-functions to update
            qlearning = qlearning1 if jax.random.bernoulli(q1.rng) else qlearning2
            metrics = qlearning.update(transition_batch)
            env.record_metrics(metrics)

            # delay policy updates
            if env.T % 2 == 0:
                metrics = determ_pg.update(transition_batch)
                env.record_metrics(metrics)

            # sync target models
            pi_targ.soft_update(pi, tau=0.01)
            q1_targ.soft_update(q1, tau=0.01)
            q2_targ.soft_update(q2, tau=0.01)

        if done or truncated:
            break

        s = s_next
