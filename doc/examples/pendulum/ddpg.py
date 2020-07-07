import os
from functools import partial

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix


# set some env vars
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'     # tell JAX to use CPU
os.environ['JAX_PLATFORM_NAME'] = 'gpu'     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/ddpg"
gifs_filepath = "./data/gifs/ddpg/T{:08d}.gif"
coax.enable_logging('ddpg')


# env with preprocessing
env = gym.make('Pendulum-v0')  # AtariPreprocessing does frame skipping
env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


class MLP(coax.FuncApprox):
    """ multi-layer perceptron with one hidden layer """
    def body(self, S, is_training):
        seq = hk.Sequential((
            coax.utils.double_relu,
            hk.Linear(6),
            partial(hk.BatchNorm(False, False, 0.95), is_training=is_training),
            coax.utils.double_relu,
            hk.Linear(6),
            partial(hk.BatchNorm(False, False, 0.95), is_training=is_training),
            coax.utils.double_relu,
        ))
        return seq(S)

    def state_action_combiner(self, X_s, X_a, is_training):
        seq = hk.Sequential((
            coax.utils.double_relu,
            hk.Linear(6),
            partial(hk.BatchNorm(False, False, 0.95), is_training=is_training),
            coax.utils.double_relu,
            hk.Linear(6),
            partial(hk.BatchNorm(False, False, 0.95), is_training=is_training),
            coax.utils.double_relu,
        ))
        return jnp.concatenate((X_s, seq(X_a)), axis=1)

    def head_pi(self, X_s, is_training):
        mu = hk.Linear(self.action_shape_flat, w_init=jnp.zeros)(X_s)
        logvar = jax.lax.stop_gradient(jnp.full_like(mu, fill_value=0.))  # DDPG can't learn var
        return {'mu': mu, 'logvar': logvar}

    def optimizer(self):
        return optix.chain(
            optix.clip_by_global_norm(1.),
            optix.adam(learning_rate=self.optimizer_kwargs.get('learning_rate', 1e-3)),
        )


# define function approximators
func = MLP(env, random_seed=13, learning_rate=1e-3)
pi = coax.Policy(func)
q = coax.Q(func)


# target network
q_targ = q.copy()
pi_targ = pi.copy()


# experience tracer
buffer = coax.ExperienceReplayBuffer(env, capacity=20000, n=5, gamma=0.9)


# policy regularizer (avoid premature exploitation)
kl_div = coax.policy_regularizers.KLDivRegularizer(pi, beta=0.001)


# value transform
log_transform = coax.value_transforms.LogTransform()


# updaters
qlearning = coax.td_learning.QLearningMode(q, pi_targ, q_targ, value_transform=log_transform)
sarsa = coax.td_learning.Sarsa(q, q_targ, value_transform=log_transform)
ddpg = coax.policy_objectives.DeterministicPG(pi, q, regularizer=kl_div)


# train
for ep in range(1000):
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi(s, return_logp=True)
        print(a, logp, pi.dist_params(s))
        s_next, r, done, info = env.step(a)

        # update
        buffer.add(s, a, r, done, logp)
        if len(buffer) >= 1000:
            transition_batch = buffer.sample(batch_size=8)
            ddpg.update(transition_batch)
            sarsa.update(transition_batch)
            # qlearning.update(transition_batch)

        # sync target networks
        if env.T % 1000 == 0:
            q_targ.smooth_update(q, tau=0.1)
            pi_targ.smooth_update(pi, tau=0.1)

        if done:
            break

        s = s_next


coax.utils.render_episode(env, pi)
