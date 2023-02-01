import coax
import jax
import jax.numpy as jnp
import gymnasium
import haiku as hk
import optax


# the MDP
env = gymnasium.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)


def func_pi(S, is_training):
    logits = hk.Linear(env.action_space.n, w_init=jnp.zeros)
    return {'logits': logits(S)}


# function approximators
pi = coax.Policy(func_pi, env)


# experience tracer
tracer = coax.reward_tracing.MonteCarlo(gamma=0.9)


# updater
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(0.01))


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
            transition_batch = tracer.pop()
            vanilla_pg.update(transition_batch, Adv=transition_batch.Rn)

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

    a = pi.mode(s)
    s, r, done, truncated, info = env.step(a)

    env.render()

    if done or truncated:
        break


if env.avg_G < env.spec.reward_threshold:
    name = globals().get('__file__', 'this script')
    raise RuntimeError(f"{name} failed to reach env.spec.reward_threshold")
