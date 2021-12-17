import time
import inspect
from abc import ABC, abstractmethod
from typing import Optional

import gym
from jax.lib.xla_bridge import get_backend

from ..typing import Policy
from ..wrappers import TrainMonitor
from ..reward_tracing._base import BaseRewardTracer
from ..experience_replay._base import BaseReplayBuffer


__all__ = (
    'Worker',
)


class WorkerError(Exception):
    pass


class Worker(ABC):
    r"""

    The base class for defining workers as part of a distributed agent.

    Parameters
    ----------
    env : gym.Env | str | function

        Specifies the gym-style environment by either passing the env itself (gym.Env), its name
        (str), or a function that generates the environment.

    param_store : Worker, optional

        A distributed agent is presumed to have one worker that plays the role of a parameter store.
        To define the parameter-store worker itself, you must leave :code:`param_store=None`. For
        other worker roles, however, :code:`param_store` must be provided.

    pi : Policy, optional

        The behavior policy that is used by rollout workers to generate experience.

    tracer : RewardTracer, optional

        The reward tracer that is used by rollout workers.

    buffer : ReplayBuffer, optional

        The experience-replay buffer that is populated by rollout workers and sampled from by
        learners.

    buffer_warmup : int, optional

        The warmup period for the experience replay buffer, i.e. the minimal number of transitions
        that need to be stored in the replay buffer before we start sampling from it.

    name : str, optional

        A human-readable identifier of the worker.

    """
    pi: Optional[Policy] = None
    tracer: Optional[BaseRewardTracer] = None
    buffer: Optional[BaseReplayBuffer] = None
    buffer_warmup: Optional[int] = None

    def __init__(
            self, env,
            param_store=None,
            pi=None,
            tracer=None,
            buffer=None,
            buffer_warmup=None,
            name=None):

        # import inline to avoid hard dependency on ray
        import ray
        import ray.actor
        self.__ray = ray

        self.env = _check_env(env, name)
        self.param_store = param_store
        self.pi = pi
        self.tracer = tracer
        self.buffer = buffer
        self.buffer_warmup = buffer_warmup
        self.name = name
        self.env.logger.info(f"JAX platform name: '{get_backend().platform}'")

    @abstractmethod
    def get_state(self):
        r"""

        Get the internal state that is shared between workers.

        Returns
        -------
        state : object

            The internal state. This will be consumed by :func:`set_state(state) <set_state>`.

        """
        pass

    @abstractmethod
    def set_state(self, state):
        r"""

        Set the internal state that is shared between workers.

        Parameters
        ----------
        state : object

            The internal state, as returned by :func:`get_state`.

        """
        pass

    @abstractmethod
    def trace(self, s, a, r, done, logp=0.0, w=1.0):
        r"""

        This implements the reward-tracing step of a single, raw transition.

        Parameters
        ----------
        s : state observation

            A single state observation.

        a : action

            A single action.

        r : float

            A single observed reward.

        done : bool

            Whether the episode has finished.

        logp : float, optional

            The log-propensity :math:`\log\pi(a|s)`.

        w : float, optional

            Sample weight associated with the given state-action pair.


        """
        pass

    @abstractmethod
    def learn(self, transition_batch):
        r""" Update the model parameters given a transition batch. """
        pass

    def rollout(self):
        assert self.pi is not None
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a, logp = self.pi(s, return_logp=True)
            s_next, r, done, info = self.env.step(a)

            self.trace(s, a, r, done, logp)

            if done:
                break

            s = s_next

    def rollout_loop(self, max_total_steps, reward_threshold=None):
        reward_threshold = _check_reward_threshold(reward_threshold, self.env)
        T_global = self.pull_getattr('env.T')
        while T_global < max_total_steps and self.env.avg_G < reward_threshold:
            self.pull_state()
            self.rollout()
            metrics = self.pull_metrics()
            metrics['throughput/rollout_loop'] = 1000 / self.env.dt_ms
            metrics['episode/T_global'] = T_global = self.pull_getattr('env.T') + self.env.t
            self.push_setattr('env.T', T_global)  # not exactly thread-safe, but that's okay
            self.env.record_metrics(metrics)

    def learn_loop(self, max_total_steps, batch_size=32):
        throughput = 0.
        while self.pull_getattr('env.T') < max_total_steps:
            t_start = time.time()
            self.pull_state()
            metrics = self.learn(self.buffer_sample(batch_size=batch_size))
            metrics['throughput/learn_loop'] = throughput
            self.push_state()
            self.push_metrics(metrics)
            throughput = batch_size / (time.time() - t_start)

    def buffer_len(self):
        if self.param_store is None:
            assert self.buffer is not None
            len_ = len(self.buffer)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            len_ = self.__ray.get(self.param_store.buffer_len.remote())
        else:
            len_ = self.param_store.buffer_len()
        return len_

    def buffer_add(self, transition_batch, Adv=None):
        if self.param_store is None:
            assert self.buffer is not None
            if 'Adv' in inspect.signature(self.buffer.add).parameters:  # duck typing
                self.buffer.add(transition_batch, Adv=Adv)
            else:
                self.buffer.add(transition_batch)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            self.__ray.get(self.param_store.buffer_add.remote(transition_batch, Adv))
        else:
            self.param_store.buffer_add(transition_batch, Adv)

    def buffer_update(self, transition_batch_idx, Adv):
        if self.param_store is None:
            assert self.buffer is not None
            self.buffer.update(transition_batch_idx, Adv=Adv)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            self.__ray.get(self.param_store.buffer_update.remote(transition_batch_idx, Adv))
        else:
            self.param_store.buffer_update(transition_batch_idx, Adv)

    def buffer_sample(self, batch_size=32):
        buffer_warmup = max(self.buffer_warmup or 0, batch_size)
        wait_secs = 1 / 1024.
        buffer_len = self.buffer_len()
        while buffer_len < buffer_warmup:
            self.env.logger.debug(
                f"buffer insufficiently populated: {buffer_len}/{buffer_warmup}; "
                f"waiting for {wait_secs}s")
            time.sleep(wait_secs)
            wait_secs = min(30, wait_secs * 2)  # wait at most 30s between tries
            buffer_len = self.buffer_len()

        if self.param_store is None:
            assert self.buffer is not None
            transition_batch = self.buffer.sample(batch_size=batch_size)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            transition_batch = self.__ray.get(
                self.param_store.buffer_sample.remote(batch_size=batch_size))
        else:
            transition_batch = self.param_store.buffer_sample(batch_size=batch_size)
        assert transition_batch is not None
        return transition_batch

    def pull_state(self):
        assert self.param_store is not None, "cannot call pull_state on param_store itself"
        if isinstance(self.param_store, self.__ray.actor.ActorHandle):
            self.set_state(self.__ray.get(self.param_store.get_state.remote()))
        else:
            self.set_state(self.param_store.get_state())

    def push_state(self):
        assert self.param_store is not None, "cannot call push_state on param_store itself"
        if isinstance(self.param_store, self.__ray.actor.ActorHandle):
            self.__ray.get(self.param_store.set_state.remote(self.get_state()))
        else:
            self.param_store.set_state(self.get_state())

    def pull_metrics(self):
        if self.param_store is None:
            metrics = self.env.get_metrics()
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            metrics = self.__ray.get(self.param_store.pull_metrics.remote()).copy()
        else:
            metrics = self.param_store.pull_metrics()
        return metrics

    def push_metrics(self, metrics):
        if self.param_store is None:
            self.env.record_metrics(metrics)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            self.__ray.get(self.param_store.push_metrics.remote(metrics))
        else:
            self.param_store.push_metrics(metrics)

    def pull_getattr(self, name, default_value=...):
        if self.param_store is None:
            value = _getattr_recursive(self, name, default_value)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            value = self.__ray.get(self.param_store.pull_getattr.remote(name, default_value))
        else:
            value = self.param_store.pull_getattr(name, default_value)
        return value

    def push_setattr(self, name, value):
        if self.param_store is None:
            _setattr_recursive(self, name, value)
        elif isinstance(self.param_store, self.__ray.actor.ActorHandle):
            self.__ray.get(self.param_store.push_setattr.remote(name, value))
        else:
            self.param_store.push_setattr(name, value)


# -- some helper functions (boilerplate) --------------------------------------------------------- #


def _check_env(env, name):
    if isinstance(env, gym.Env):
        pass
    elif isinstance(env, str):
        env = gym.make(env)
    elif hasattr(env, '__call__'):
        env = env()
    else:
        raise TypeError(f"env must be a gym.Env, str or callable; got: {type(env)}")

    if getattr(getattr(env, 'spec', None), 'max_episode_steps', None) is None:
        raise ValueError(
            "env.spec.max_episode_steps not set; please register env with "
            "gym.register('Foo-v0', entry_point='foo.Foo', max_episode_steps=...) "
            "or wrap your env with: env = gym.wrappers.TimeLimit(env, max_episode_steps=...)")

    if not isinstance(env, TrainMonitor):
        env = TrainMonitor(env, name=name, log_all_metrics=True)

    return env


def _check_reward_threshold(reward_threshold, env):
    if reward_threshold is None:
        reward_threshold = getattr(getattr(env, 'spec', None), 'reward_threshold', None)
    if reward_threshold is None:
        reward_threshold = float('inf')
    return reward_threshold


def _getattr_recursive(obj, name, default=...):
    if '.' not in name:
        return getattr(obj, name) if default is Ellipsis else getattr(obj, name, default)

    name, subname = name.split('.', 1)
    return _getattr_recursive(getattr(obj, name), subname, default)


def _setattr_recursive(obj, name, value):
    if '.' not in name:
        return setattr(obj, name, value)

    name, subname = name.split('.', 1)
    return _setattr_recursive(getattr(obj, name), subname, value)
