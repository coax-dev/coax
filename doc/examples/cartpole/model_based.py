import coax
import gymnasium
import jax.numpy as jnp
import haiku as hk
import optax
from coax.value_losses import mse


# the name of this script
name = 'model_based'

# the cart-pole MDP
env = gymnasium.make('CartPole-v0', render_mode='rgb_array')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func_v(S, is_training):
    potential = hk.Sequential((jnp.square, hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    return -jnp.square(S[:, 3]) + potential(S[:, :3])  # kinetic term is angular velocity squared


def func_p(S, A, is_training):
    dS = hk.Linear(4, w_init=jnp.zeros)
    return S + dS(A)


def func_r(S, A, is_training):
    return jnp.ones(S.shape[0])  # CartPole yields r=1 at every time step (no need to learn)


# function approximators
p = coax.TransitionModel(func_p, env)
v = coax.V(func_v, env, observation_preprocessor=p.observation_preprocessor)
r = coax.RewardFunction(func_r, env, observation_preprocessor=p.observation_preprocessor)


# composite objects
q = coax.SuccessorStateQ(v, p, r, gamma=0.9)
pi = coax.EpsilonGreedy(q, epsilon=0.)  # no exploration


# reward tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=q.gamma)


# updaters
adam = optax.chain(optax.apply_every(k=16), optax.adam(1e-4))
simple_td = coax.td_learning.SimpleTD(v, loss_function=mse, optimizer=adam)

sgd = optax.sgd(1e-3, momentum=0.9, nesterov=True)
model_updater = coax.model_updaters.ModelUpdater(p, optimizer=sgd)


while env.T < 100000:
    s, info = env.reset()
    env.render()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, truncated, info = env.step(a)
        env.render()

        tracer.add(s, a, r, done or truncated)
        while tracer:
            transition_batch = tracer.pop()
            env.record_metrics(simple_td.update(transition_batch))
            env.record_metrics(model_updater.update(transition_batch))

        if done or truncated:
            break

        s = s_next

    # early stopping
    if env.ep >= 5 and env.avg_G > env.spec.reward_threshold:
        break


# run env one more time to render
coax.utils.generate_gif(env, policy=pi, filepath=f"./data/{name}.gif", duration=25)
