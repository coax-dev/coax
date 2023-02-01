import coax
import gymnasium
import jax
import jax.numpy as jnp
import haiku as hk
import optax


# the MDP
env = gymnasium.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)


def func_pi(S, is_training):
    logits = hk.Linear(env.action_space.n, w_init=jnp.zeros)
    return {'logits': logits(S)}


def func_q(S, A, is_training):
    value = hk.Sequential((hk.Flatten(), hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    X = jax.vmap(jnp.kron)(S, A)  # S and A are one-hot encoded
    return value(X)


# function approximators
pi = coax.Policy(func_pi, env)
q1 = coax.Q(func_q, env)
q2 = coax.Q(func_q, env)


# target networks
q1_targ = q1.copy()
q2_targ = q2.copy()
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=128)


# updaters
determ_pg = coax.policy_objectives.DeterministicPG(pi, q1_targ, optimizer=optax.adam(0.02))
qlearning1 = coax.td_learning.ClippedDoubleQLearning(
    q1, q_targ_list=[q1_targ, q2_targ], optimizer=optax.adam(0.02))
qlearning2 = coax.td_learning.ClippedDoubleQLearning(
    q2, q_targ_list=[q1_targ, q2_targ], optimizer=optax.adam(0.02))


# train
for ep in range(500):
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # small incentive to keep moving
        if jnp.array_equal(s_next, s):
            r = -0.01

        # update
        tracer.add(s, a, r, done or truncated)
        while tracer:
            buffer.add(tracer.pop())

        if len(buffer) == buffer.capacity:
            transition_batch = buffer.sample(batch_size=16)

            # flip a coin to decide which of the q-functions to update
            qlearning = qlearning1 if jax.random.bernoulli(q1.rng) else qlearning2
            qlearning.update(transition_batch)

            # delayed policy updates
            if env.T % 2 == 0:
                determ_pg.update(transition_batch)

            # sync copies
            q1_targ.soft_update(q1, tau=0.1)
            q2_targ.soft_update(q2, tau=0.1)
            pi_targ.soft_update(pi, tau=0.1)

        if done or truncated:
            break

        s = s_next

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        break


# run env one more time to render
s, info = env.reset()
env.render()

for t in range(env.spec.max_episode_steps):

    # print individual action probabilities
    params = pi.dist_params(s)
    propensities = jax.nn.softmax(params['logits'])
    for i, p in enumerate(propensities):
        print("  π({:s}|s) = {:.3f}".format('LDRU'[i], p))
    for i, q_ in enumerate(q1(s)):
        print("  q1(s,{:s}) = {:.3f}".format('LDRU'[i], q_))
    for i, q_ in enumerate(q2(s)):
        print("  q2(s,{:s}) = {:.3f}".format('LDRU'[i], q_))

    a = pi.mode(s)
    s, r, done, truncated, info = env.step(a)

    env.render()

    if done or truncated:
        break


if env.avg_G < env.spec.reward_threshold:
    name = globals().get('__file__', 'this script')
    raise RuntimeError(f"{name} failed to reach env.spec.reward_threshold")
