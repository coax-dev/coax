import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/ppo_standard"
gifs_filepath = "./data/gifs/ppo/T{:08d}.gif"
coax.enable_logging('ppo')


# env with preprocessing
env = gym.make('Pendulum-v0')  # AtariPreprocessing does frame skipping
env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


class Func(coax.FuncApprox):
    def body(self, S, is_training):
        return S

    def head_pi(self, X_s, is_training):
        mu = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(jnp.prod(env.action_space.shape)),
            hk.Reshape(env.action_space.shape),
        ))
        logvar = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(jnp.prod(env.action_space.shape)),
            hk.Reshape(env.action_space.shape),
        ))
        return {'mu': mu(X_s), 'logvar': logvar(X_s)}

    def head_v(self, X_s, is_training):
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1),
        ))
        return seq(X_s)


# define function approximators
func = Func(env, random_seed=13, learning_rate=1e-3)
pi = coax.Policy(func)
v = coax.V(func)


# target network
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStepCache(n=5, gamma=0.99)


# policy regularizer (avoid premature exploitation)
policy_reg = coax.policy_regularizers.EntropyRegularizer(pi, beta=1)


# updaters
value_td = coax.td_learning.ValueTD(v, loss_function=coax.value_losses.mse)
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=policy_reg)


# train
while env.T < 1000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_targ(s, return_logp=True)
        # print(a, logp, pi.dist_params(s))
        s_next, r, done, info = env.step(a)

        # trace rewards
        tracer.add(s, a, r, done, logp)

        if done:
            transition_batch = tracer.flush()
            for _ in range(4):
                Adv = value_td.td_error(transition_batch)
                metrics = {}
                metrics.update(ppo_clip.update(transition_batch, Adv))
                metrics.update(value_td.update(transition_batch))
                env.record_metrics(metrics)

            pi_targ.soft_update(pi, tau=0.001)

            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(env=env, policy=pi.greedy, filepath=gifs_filepath.format(T))

    # stop early
    if env.ep > 100 and env.avg_G > -150:
        break
