import gymnasium
import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax


# the name of this script
name = 'ppo'

# the Pendulum MDP
env = gymnasium.make('Pendulum-v1', render_mode='rgb_array')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func_pi(S, is_training):
    shared = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
    ))
    mu = hk.Sequential((
        shared,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    logvar = hk.Sequential((
        shared,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape),
    ))
    return {'mu': mu(S), 'logvar': logvar(S)}


def func_v(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return seq(S)


# define function approximators
pi = coax.Policy(func_pi, env)
v = coax.V(func_v, env)


# target network
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=512)


# policy regularizer (avoid premature exploitation)
policy_reg = coax.regularizers.EntropyRegularizer(pi, beta=0.01)


# updaters
simpletd = coax.td_learning.SimpleTD(v, optimizer=optax.adam(1e-3))
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=policy_reg, optimizer=optax.adam(1e-4))


# train
while env.T < 1000000:
    s, info = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_targ(s, return_logp=True)
        s_next, r, done, truncated, info = env.step(a)

        # trace rewards
        tracer.add(s, a, r, done or truncated, logp)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= buffer.capacity:
            for _ in range(int(4 * buffer.capacity / 32)):  # 4 passes per round
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = simpletd.update(transition_batch, return_td_error=True)
                metrics_pi = ppo_clip.update(transition_batch, td_error)
                env.record_metrics(metrics_v)
                env.record_metrics(metrics_pi)

            buffer.clear()
            pi_targ.soft_update(pi, tau=0.1)

        if done or truncated:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000  # round to 10000s
        coax.utils.generate_gif(
            env=env, policy=pi, filepath=f"./data/gifs/{name}/T{T:08d}.gif")
