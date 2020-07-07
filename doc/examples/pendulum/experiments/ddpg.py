import os
import argparse

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--max_num_episodes', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--logvar', type=float, required=True)
parser.add_argument('--log_transform', action='store_true')
parser.add_argument('--kl_div_beta', type=float, required=True)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--sync_tau', type=float, required=True)
parser.add_argument('--sync_period', type=int, required=True)
args = parser.parse_args()

experiment_id = (
    'ddpg'
    f'|max_num_episodes={args.max_num_episodes}'
    f'|learning_rate={args.learning_rate}'
    f'|logvar={args.logvar}'
    f'|log_transform={args.log_transform}'
    f'|kl_div_beta={args.kl_div_beta}'
    f'|gamma={args.gamma}'
    f'|n_step={args.n_step}'
    f'|sync_tau={args.sync_tau}'
    f'|sync_period={args.sync_period}'
)


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')     # tell JAX to use CPU
# os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = f"./data/tensorboard/{experiment_id}"
gifs_filepath = f"./data/gifs/{experiment_id}/T{{:08d}}.gif"
coax.enable_logging(experiment_id)


# env with preprocessing
env = gym.make('Pendulum-v0')  # AtariPreprocessing does frame skipping
env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


class MLP(coax.FuncApprox):
    """ multi-layer perceptron with one hidden layer """
    def __init__(self, *args, logvar=0., **kwargs):
        self.logvar = float(logvar)
        super().__init__(*args, **kwargs)

    def body(self, S, is_training):
        seq = hk.Sequential((
            lambda x: jnp.stack((x, jnp.square(x)), axis=1),
            hk.Linear(6), jnp.tanh,
            hk.Linear(6), jnp.tanh,
        ))
        return seq(S)

    def head_pi(self, X_s, is_training):
        mu = hk.Linear(self.action_shape_flat, w_init=jnp.zeros)(X_s)
        # DDPG can't learn variance
        logvar = jnp.full_like(mu, fill_value=self.logvar)
        return {'mu': mu, 'logvar': logvar}


# define function approximators
func = MLP(
    env, random_seed=13, learning_rate=args.kl_div_beta, logvar=args.logvar)
pi = coax.Policy(func)
q = coax.Q(func)


# target network
q_targ = q.copy()
pi_targ = pi.copy()


# experience tracer
cache = coax.NStepCache(env, n=args.n_step, gamma=args.gamma)


# policy regularizer (avoid premature exploitation)
kl_div = coax.policy_regularizers.KLDivRegularizer(pi, beta=args.kl_div_beta)


# value transform
if args.log_transform:
    transform = coax.value_transforms.LogTransform()
else:
    transform = None


# updaters
sarsa = coax.td_learning.Sarsa(q, q_targ, value_transform=transform)
ddpg = coax.policy_objectives.DeterministicPG(pi, q, regularizer=kl_div)


# train
for ep in range(args.max_num_episodes):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # update
        cache.add(s, a, r, done)
        while cache:
            transition_batch = cache.pop()
            ddpg.update(transition_batch)
            sarsa.update(transition_batch)

        # sync copies
        if env.T % args.sync_period == 0:
            q_targ.smooth_update(q, tau=args.sync_tau)
            pi_targ.smooth_update(pi, tau=args.sync_tau)

        if done:
            break

        s = s_next


coax.utils.render_episode(env, pi)
