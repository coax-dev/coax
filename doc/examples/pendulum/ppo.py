import os
from functools import partial

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix


# set some env vars
os.environ['JAX_PLATFORM_NAME'] = 'cpu'     # tell JAX to use CPU
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'     # tell JAX to use GPU
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


class MLP(coax.FuncApprox):
    """ multi-layer perceptron with one hidden layer """
    def body(self, S, is_training):
        seq = hk.Sequential((
            lambda x: jnp.concatenate((x, jnp.square(x)), axis=-1),
            hk.Linear(6), jnp.tanh,
            # partial(hk.BatchNorm(False, False, 0.95), is_training=is_training),
            hk.Linear(6), jnp.tanh,
            # partial(hk.BatchNorm(False, False, 0.95), is_training=is_training),
        ))
        return seq(S)

    # def head_pi(self, X_s, is_training):
    #     mu = hk.Linear(self.action_shape_flat, w_init=jnp.zeros)(X_s)
    #     logvar = jax.lax.stop_gradient(jnp.full_like(mu, fill_value=0.))  # don't learn var
    #     return {'mu': mu, 'logvar': logvar}

    def optimizer(self):
        learning_rate = self.optimizer_kwargs.get('learning_rate', 1e-3)
        return optix.chain(
            optix.clip_by_global_norm(1.),
            optix.adam(learning_rate),
        )


# define function approximators
func = MLP(env, random_seed=13, learning_rate=1e-3)
pi = coax.Policy(func)
v = coax.V(func)


# target network
pi_targ = pi.copy()


# experience tracer
buffer = coax.ExperienceReplayBuffer(env, capacity=512, n=5, gamma=0.9)


# policy regularizer (avoid premature exploitation)
policy_reg = coax.policy_regularizers.EntropyRegularizer(pi, beta=0.01)


# value transform
log_transform = None  #coax.value_transforms.LogTransform()


# updaters
value_td = coax.td_learning.ValueTD(v, value_transform=log_transform)
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=policy_reg)


# train
while env.T < 1000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_targ(s, return_logp=True)
        # print(a, logp, pi.dist_params(s))
        s_next, r, done, info = env.step(a)

        # update
        buffer.add(s, a, r, done, logp)
        if len(buffer) >= buffer.capacity:
            batch_size = 32
            for _ in range(int(4 * len(buffer) / batch_size)):
                transition_batch = buffer.sample(batch_size)
                Adv = value_td.td_error(transition_batch)

                metrics = {}
                metrics.update(ppo_clip.update(transition_batch, Adv))
                metrics.update(value_td.update(transition_batch))
                env.record_metrics(metrics)

            buffer.clear()
            pi_targ.smooth_update(pi, tau=0.1)

        if done:
            break

        s = s_next

    # stop early
    if env.ep > 100 and env.avg_G > -150:
        break

coax.utils.render_episode(env, pi)
