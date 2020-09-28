import coax
import gym
import jax
import jax.numpy as jnp
import haiku as hk
import optax


# the MDP
env = gym.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)


def func_pi(S, is_training):
    logits = hk.Linear(env.action_space.n, w_init=jnp.zeros)
    S = hk.one_hot(S, env.observation_space.n)
    return {'logits': logits(S)}


def func_q(S, A, is_training):
    value = hk.Sequential((hk.Flatten(), hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    S = hk.one_hot(S, env.observation_space.n)
    X = jax.vmap(jnp.kron)(S, A)  # A is already one-hot encoded
    return value(X)


# function approximators
pi = coax.Policy(func_pi, env)
q = coax.Q(func_q, env)


# target networks
q_targ = q.copy()
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=128)


# updaters
qlearning = coax.td_learning.QLearning(q, pi_targ, q_targ, optimizer=optax.adam(0.02))
determ_pg = coax.policy_objectives.DeterministicPG(pi, q, optimizer=optax.adam(0.01))


# train
for ep in range(500):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # small incentive to keep moving
        if jnp.array_equal(s_next, s):
            r = -0.01

        # update
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        if len(buffer) == buffer.capacity:
            transition_batch = buffer.sample(batch_size=16)
            determ_pg.update(transition_batch)
            qlearning.update(transition_batch)

            # sync copies
            q_targ.soft_update(q, tau=0.01)
            pi_targ.soft_update(pi, tau=0.01)

        if done:
            break

        s = s_next


# run env one more time to render
s = env.reset()
env.render()

for t in range(env.spec.max_episode_steps):

    # print individual action probabilities
    params = pi.dist_params(s)
    propensities = jax.nn.softmax(params['logits'])
    for i, p in enumerate(propensities):
        print("  π({:s}|s) = {:.3f}".format('LDRU'[i], p))
    for i, q_ in enumerate(q(s)):
        print("  q(s,{:s}) = {:.3f}".format('LDRU'[i], q_))

    a = pi.mode(s)
    s, r, done, info = env.step(a)

    env.render()

    if done:
        break
