import os

import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from jax.experimental import optix
from ray.rllib.env.atari_wrappers import wrap_deepmind


# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'              # tell XLA to be quiet


# filepaths etc
tensorboard_dir = "./data/tensorboard/ppo"
gifs_filepath = "./data/gifs/ppo/T{:08d}.gif"
coax.enable_logging('ppo')


# env with preprocessing
env = gym.make('PongNoFrameskip-v4')  # wrap_deepmind will do frame skipping
env = wrap_deepmind(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir=tensorboard_dir)


class Func(coax.FuncApprox):
    def body(self, S, is_training):
        M = coax.utils.diff_transform_matrix(num_frames=S.shape[-1])
        seq = hk.Sequential([  # S.shape = [batch, h, w, num_stack]
            lambda x: jnp.dot(S / 255, M),  # [b, h, w, n]
            hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
            hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256), jax.nn.relu,
        ])
        return seq(S)

    def optimizer(self):
        return optix.adam(learning_rate=0.00025)


# function approximators (using two separate function approximators)
func_pi, func_v = Func(env), Func(env)
pi = coax.Policy(func_pi)
pi_old = pi.copy()
v = coax.V(func_v)
v_targ = v.copy()

# we'll use this to temporarily store our experience
tracer = coax.reward_tracing.NStepCache(n=5, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)

# policy regularizer (avoid premature exploitation)
kl_div = coax.policy_regularizers.KLDivRegularizer(pi, beta=0.001)

# updaters
value_td = coax.td_learning.ValueTD(v, v_targ)
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=kl_div)


# run episodes
while env.T < 3000000:
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a, logp = pi_old(s, return_logp=True)
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= buffer.capacity:
            num_batches = int(4 * buffer.capacity / 32)  # 4 epochs per round
            for _ in range(num_batches):
                transition_batch = buffer.sample(32)

                Adv = value_td.td_error(transition_batch)
                metrics_pi = ppo_clip.update(transition_batch, Adv)
                metrics_v = value_td.update(transition_batch)

                metrics = coax.utils.merge_dicts(metrics_pi, metrics_v)
                env.record_metrics(metrics)

            buffer.clear()

            # sync target networks
            pi_old.soft_update(pi, tau=0.1)
            v_targ.soft_update(v, tau=0.1)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 50000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(
            env=env, policy=pi.greedy, resize_to=(320, 420),
            filepath=gifs_filepath.format(T))
