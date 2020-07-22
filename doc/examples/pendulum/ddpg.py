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
tensorboard_dir = "./data/tensorboard/ddpg"
gifs_filepath = "./data/gifs/ddpg/T{:08d}.gif"
coax.enable_logging('ddpg')


# env with preprocessing
env = gym.make('Pendulum-v0')  # AtariPreprocessing does frame skipping
env = coax.wrappers.BoxActionsToReals(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir)


class Func(coax.FuncApprox):
    def body(self, S, is_training):
        return S  # trivial

    def head_pi(self, S, is_training):
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(jnp.prod(env.action_space.shape)),
            hk.Reshape(env.action_space.shape),
        ))
        mu = seq(S)
        return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}  # (almost) deterministic

    def state_action_combiner(self, S, A, is_training):
        flatten = hk.Flatten()
        return jnp.concatenate([flatten(S), jnp.tanh(flatten(A))], axis=1)

    def head_q1(self, X_sa, is_training):
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1),
        ))
        return seq(X_sa)


# main function approximators
func = Func(env, learning_rate=1e-3)
pi = coax.Policy(func)
q = coax.Q(func)


# target networks
pi_targ = pi.copy()
q_targ = q.copy()


# target network
q_targ = q.copy()
pi_targ = pi.copy()


# experience tracer
tracer = coax.reward_tracing.NStepCache(n=5, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=25000)


# updaters
qlearning = coax.td_learning.QLearningMode(q, pi_targ, q_targ, loss_function=coax.value_losses.mse)
determ_pg = coax.policy_objectives.DeterministicPG(pi, q_targ)


# action noise
noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=0.2, theta=0.15)


# train
while env.T < 1000000:
    s = env.reset()
    noise.reset()
    noise.sigma *= 0.99  # slowly decrease noise scale

    for t in range(env.spec.max_episode_steps):
        a = noise(pi(s))
        # print(a, pi.dist_params(s))
        s_next, r, done, info = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 5000:
            transition_batch = buffer.sample(batch_size=1024)

            metrics = {'OrnsteinUhlenbeckNoise/sigma': noise.sigma}
            metrics.update(determ_pg.update(transition_batch))
            metrics.update(qlearning.update(transition_batch))
            env.record_metrics(metrics)

            # sync target networks
            q_targ.soft_update(q, tau=0.001)
            pi_targ.soft_update(pi, tau=0.001)

        if done:
            break

        s = s_next

    # generate an animated GIF to see what's going on
    if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(env=env, policy=pi.greedy, filepath=gifs_filepath.format(T))
