import os

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem

import gym
import ray
import jax
import jax.numpy as jnp
import coax
import haiku as hk
import optax


# name of this script
name, _ = os.path.splitext(os.path.basename(__file__))


@ray.remote(num_cpus=1, num_gpus=0)
class ApexWorker(coax.Worker):
    def __init__(self, name, param_store=None, tensorboard_dir=None):
        env = make_env(name, tensorboard_dir)

        # function approximator
        self.q = coax.Q(forward_pass, env)
        self.q_targ = self.q.copy()

        # tracer and updater
        self.q_updater = coax.td_learning.QLearning(
            self.q, q_targ=self.q_targ, optimizer=optax.adam(3e-4))

        # schedule for beta parameter used in PrioritizedReplayBuffer
        self.buffer_beta = coax.utils.StepwiseLinearFunction((0, 0.4), (1000000, 1))

        super().__init__(
            env=env,
            param_store=param_store,
            pi=coax.BoltzmannPolicy(self.q, temperature=0.015),
            tracer=coax.reward_tracing.NStep(n=1, gamma=0.99),
            buffer=(
                coax.experience_replay.PrioritizedReplayBuffer(capacity=1000000, alpha=0.6)
                if param_store is None else None),
            buffer_warmup=50000,
            name=name)

    def get_state(self):
        return self.q.params, self.q.function_state, self.q_targ.params, self.q_targ.function_state

    def set_state(self, state):
        self.q.params, self.q.function_state, self.q_targ.params, self.q_targ.function_state = state

    def trace(self, s, a, r, done, logp):
        self.tracer.add(s, a, r, done, logp)
        if done:
            transition_batch = self.tracer.flush()
            for chunk in coax.utils.chunks_pow2(transition_batch):
                td_error = self.q_updater.td_error(chunk)
                self.buffer_add(chunk, td_error)

    def learn(self, transition_batch):
        metrics, td_error = self.q_updater.update(transition_batch, return_td_error=True)
        self.buffer_update(transition_batch.idx, td_error)
        self.q_targ.soft_update(self.q, tau=0.001)
        self.push_setattr('buffer.beta', self.buffer_beta(self.env.T))
        return metrics


def make_env(name=None, tensorboard_dir=None):
    env = gym.make('PongNoFrameskip-v4')  # AtariPreprocessing will do frame skipping
    env = gym.wrappers.AtariPreprocessing(env)
    env = coax.wrappers.FrameStacking(env, num_frames=3)
    env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=tensorboard_dir)
    env.spec.reward_threshold = 19.
    return env


def forward_pass(S, is_training):
    seq = hk.Sequential((
        coax.utils.diff_transform,
        hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
        hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
        hk.Flatten(),
        hk.Linear(256), jax.nn.relu,
        hk.Linear(make_env().action_space.n, w_init=jnp.zeros),
    ))
    X = jnp.stack(S, axis=-1) / 255.  # stack frames
    return seq(X)


# settings
num_actors = 6


# start ray cluster
ray.init(num_cpus=(2 + num_actors), num_gpus=0)


# the central parameter store
param_store = ApexWorker.remote('param_store')


# concurrent rollout workers
actors = [
    ApexWorker.remote(f'actor_{i}', param_store, f'data/tensorboard/apex_dqn/actor_{i}')
    for i in range(num_actors)]


# one learner
learner = ApexWorker.remote('learner', param_store)


# block until one of the remote processes terminates
ray.wait([
    learner.learn_loop.remote(max_total_steps=3000000),
    *(actor.rollout_loop.remote(max_total_steps=3000000) for actor in actors)
])
