import coax
import gymnasium
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from coax.value_losses import mse


# the name of this script
name = 'a2c'

# the cart-pole MDP
env = gymnasium.make('CartPole-v0', render_mode='rgb_array')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func_pi(S, is_training):
    logits = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros)
    ))
    return {'logits': logits(S)}


def func_v(S, is_training):
    value = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return value(S)


# these optimizers collect batches of grads before applying updates
optimizer_v = optax.chain(optax.apply_every(k=32), optax.adam(0.002))
optimizer_pi = optax.chain(optax.apply_every(k=32), optax.adam(0.001))


# value function and its derived policy
v = coax.V(func_v, env)
pi = coax.Policy(func_pi, env)

# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)

# updaters
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optimizer_pi)
simple_td = coax.td_learning.SimpleTD(v, loss_function=mse, optimizer=optimizer_v)


# train
for ep in range(1000):
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)

        # extend last reward as asymptotic best-case return
        if truncated:
            r = 1 / (1 - tracer.gamma)  # gamma + gamma^2 + gamma^3 + ... = 1 / (1 - gamma)

        tracer.add(s, a, r, done or truncated)
        while tracer:
            transition_batch = tracer.pop()
            metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
            metrics_pi = vanilla_pg.update(transition_batch, td_error)
            env.record_metrics(metrics_v)
            env.record_metrics(metrics_pi)

        if done or truncated:
            break

        s = s_next

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        break


# run env one more time to render
coax.utils.generate_gif(env, policy=pi, filepath=f"./data/{name}.gif", duration=25)
