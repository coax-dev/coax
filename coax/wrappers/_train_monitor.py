import os
import re
import datetime
import time
from collections import deque
from typing import Mapping

import numpy as np
import lz4.frame
import cloudpickle as pickle
from gym import Wrapper
from gym.spaces import Discrete
from tensorboardX import SummaryWriter

from .._base.mixins import LoggerMixin
from ..utils import enable_logging


__all__ = (
    'TrainMonitor',
)


class StreamingSample:
    def __init__(self, maxlen, random_seed=None):
        self._deque = deque(maxlen=maxlen)
        self._count = 0
        self._rnd = np.random.RandomState(random_seed)

    def reset(self):
        self._deque = deque(maxlen=self.maxlen)
        self._count = 0

    def append(self, obj):
        self._count += 1
        if len(self) < self.maxlen:
            self._deque.append(obj)
        elif self._rnd.rand() < self.maxlen / self._count:
            i = self._rnd.randint(self.maxlen)
            self._deque[i] = obj

    @property
    def values(self):
        return list(self._deque)  # shallow copy

    @property
    def maxlen(self):
        return self._deque.maxlen

    def __len__(self):
        return len(self._deque)

    def __bool__(self):
        return bool(self._deque)


class TrainMonitor(Wrapper, LoggerMixin):
    r"""
    Environment wrapper for monitoring the training process.

    This wrapper logs some diagnostics at the end of each episode and it also gives us some handy
    attributes (listed below).

    Parameters
    ----------
    env : gym environment

        A gym environment.

    tensorboard_dir : str, optional

        If provided, TrainMonitor will log all diagnostics to be viewed in tensorboard. To view
        these, point tensorboard to the same dir:

        .. code:: bash

            $ tensorboard --logdir {tensorboard_dir}

    tensorboard_write_all : bool, optional

        You may record your training metrics using the :attr:`record_metrics` method. Setting the
        ``tensorboard_write_all`` specifies whether to pass the metrics on to tensorboard
        immediately (``True``) or to wait and average them across the episode (``False``). The
        default setting (``False``) prevents tensorboard from being fluided by logs.

    log_all_metrics : bool, optional

        Whether to log all metrics. If ``log_all_metrics=False``, only a reduced set of metrics are
        logged.

    smoothing : positive int, optional

        The number of observations for smoothing the metrics. We use the following smooth update
        rule:

        .. math::

            n\ &\leftarrow\ \min(\text{smoothing}, n + 1) \\
            x_\text{avg}\ &\leftarrow\ x_\text{avg}
                + \frac{x_\text{obs} - x_\text{avg}}{n}

    \*\*logger_kwargs

        Keyword arguments to pass on to :func:`coax.utils.enable_logging`.

    Attributes
    ----------
    T : positive int

        Global step counter. This is not reset by ``env.reset()``, use ``env.reset_global()``
        instead.

    ep : positive int

        Global episode counter. This is not reset by ``env.reset()``, use ``env.reset_global()``
        instead.

    t : positive int

        Step counter within an episode.

    G : float

        The return, i.e. amount of reward accumulated from the start of the current episode.

    avg_G : float

        The average return G, averaged over the past 100 episodes.

    dt_ms : float

        The average wall time of a single step, in milliseconds.

    """
    _COUNTER_ATTRS = (
        'T', 'ep', 't', 'G', 'avg_G', '_n_avg_G', '_ep_starttime', '_ep_metrics', '_ep_actions',
        '_tensorboard_dir', '_period')

    def __init__(
            self, env,
            tensorboard_dir=None,
            tensorboard_write_all=False,
            log_all_metrics=False,
            smoothing=10,
            **logger_kwargs):

        super().__init__(env)
        self.log_all_metrics = log_all_metrics
        self.tensorboard_write_all = tensorboard_write_all
        self.smoothing = float(smoothing)
        self.reset_global()
        enable_logging(**logger_kwargs)
        self._init_tensorboard(tensorboard_dir)

    def reset_global(self):
        r""" Reset the global counters, not just the episodic ones. """
        self.T = 0
        self.ep = 0
        self.t = 0
        self.G = 0.0
        self.avg_G = 0.0
        self._n_avg_G = 0.0
        self._ep_starttime = time.time()
        self._ep_metrics = {}
        self._ep_actions = StreamingSample(maxlen=1000)
        self._period = {'T': {}, 'ep': {}}

    def reset(self):
        # write logs from previous episode:
        if self.ep:
            self._write_episode_logs()

        # increment global counters:
        self.T += 1
        self.ep += 1

        # reset episodic counters:
        self.t = 0
        self.G = 0.0
        self._ep_starttime = time.time()
        self._ep_metrics = {}
        self._ep_actions.reset()

        return self.env.reset()

    @property
    def dt_ms(self):
        if self.t <= 0:
            return np.nan
        return 1000 * (time.time() - self._ep_starttime) / self.t

    @property
    def avg_r(self):
        if self.t <= 0:
            return np.nan
        return self.G / self.t

    def step(self, a):
        self._ep_actions.append(a)
        s_next, r, done, info = self.env.step(a)
        if info is None:
            info = {}
        info['monitor'] = {'T': self.T, 'ep': self.ep}
        self.t += 1
        self.T += 1
        self.G += r
        if done:
            if self._n_avg_G < self.smoothing:
                self._n_avg_G += 1.
            self.avg_G += (self.G - self.avg_G) / self._n_avg_G

        return s_next, r, done, info

    def record_metrics(self, metrics):
        r"""
        Record metrics during the training process.

        These are used to print more diagnostics.

        Parameters
        ----------
        metrics : dict

            A dict of metrics, of type ``{name <str>: value <float>}``.

        """
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a Mapping")

        # write metrics to tensoboard
        if self.tensorboard is not None and self.tensorboard_write_all:
            for name, metric in metrics.items():
                self.tensorboard.add_scalar(
                    str(name), float(metric), global_step=self.T)

        # compute episode averages
        for k, v in metrics.items():
            if k not in self._ep_metrics:
                self._ep_metrics[k] = v, 1.
            else:
                x, n = self._ep_metrics[k]
                self._ep_metrics[k] = x + v, n + 1

    def get_metrics(self):
        r"""
        Return the current state of the metrics.

        Returns
        -------
        metrics : dict

            A dict of metrics, of type ``{name <str>: value <float>}``.

        """
        return {k: float(x) / n for k, (x, n) in self._ep_metrics.items()}

    def period(self, name, T_period=None, ep_period=None):
        if T_period is not None:
            T_period = int(T_period)
            assert T_period > 0
            if name not in self._period['T']:
                self._period['T'][name] = 1
            if self.T >= self._period['T'][name] * T_period:
                self._period['T'][name] += 1
                return True or self.period(name, None, ep_period)
            return self.period(name, None, ep_period)
        if ep_period is not None:
            ep_period = int(ep_period)
            assert ep_period > 0
            if name not in self._period['ep']:
                self._period['ep'][name] = 1
            if self.ep >= self._period['ep'][name] * ep_period:
                self._period['ep'][name] += 1
                return True
        return False

    @property
    def tensorboard(self):
        if not hasattr(self, '_tensorboard'):
            assert self._tensorboard_dir is not None
            self._tensorboard = SummaryWriter(self._tensorboard_dir)
        return self._tensorboard

    def _init_tensorboard(self, tensorboard_dir):
        if tensorboard_dir is None:
            self._tensorboard_dir = None
            self._tensorboard = None
            return

        # append timestamp to disambiguate instances
        if not re.match(r'.*/\d{8}_\d{6}$', tensorboard_dir):
            tensorboard_dir = os.path.join(
                tensorboard_dir,
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

        # only set/update if necessary
        if tensorboard_dir != getattr(self, '_tensorboard_dir', None):
            self._tensorboard_dir = tensorboard_dir
            if hasattr(self, '_tensorboard'):
                del self._tensorboard

    def _write_episode_logs(self):
        metrics = (
            f'{k:s}: {float(x) / n:.3g}'
            for k, (x, n) in self._ep_metrics.items() if (
                self.log_all_metrics
                or str(k).endswith('/loss')
                or str(k).endswith('/entropy')
                or str(k).endswith('/kl_div')
                or str(k).startswith('throughput/')
            )
        )
        self.logger.info(
            ',\t'.join((
                f'ep: {self.ep:d}',
                f'T: {self.T:,d}',
                f'G: {self.G:.3g}',
                f'avg_r: {self.avg_r:.3g}',
                f'avg_G: {self.avg_G:.3g}',
                f't: {self.t:d}',
                f'dt: {self.dt_ms:.3f}ms',
                *metrics)))

        if self.tensorboard is not None:
            metrics = {
                'episode/episode': self.ep,
                'episode/avg_reward': self.avg_r,
                'episode/return': self.G,
                'episode/steps': self.t,
                'episode/avg_step_duration_ms': self.dt_ms}
            for name, metric in metrics.items():
                self.tensorboard.add_scalar(
                    str(name), float(metric), global_step=self.T)
            if self._ep_actions:
                if isinstance(self.action_space, Discrete):
                    bins = np.arange(self.action_space.n + 1)
                else:
                    bins = 'auto'  # see also: np.histogram_bin_edges.__doc__
                self.tensorboard.add_histogram(
                    tag='actions', values=self._ep_actions.values, global_step=self.T, bins=bins)
            if self._ep_metrics and not self.tensorboard_write_all:
                for k, (x, n) in self._ep_metrics.items():
                    self.tensorboard.add_scalar(str(k), float(x) / n, global_step=self.T)
            self.tensorboard.flush()

    def __getstate__(self):
        state = self.__dict__.copy()   # shallow copy
        if '_tensorboard' in state:
            del state['_tensorboard']  # remove reference to non-pickleable attr
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_tensorboard(state['_tensorboard_dir'])

    def get_counters(self):
        r"""

        Get the current state of all internal counters.

        Returns
        -------
        counter : dict

            The dict that contains the counters.

        """
        return {k: getattr(self, k) for k in self._COUNTER_ATTRS}

    def set_counters(self, counters):
        r"""

        Restore the state of all internal counters.

        Parameters
        ----------
        counter : dict

            The dict that contains the counters.

        """
        if not (isinstance(counters, dict) and set(counters) == set(self._COUNTER_ATTRS)):
            raise TypeError(f"invalid counters dict: {counters}")
        self.__setstate__(counters)

    def save_counters(self, filepath):
        r"""

        Store the current state of all internal counters.

        Parameters
        ----------
        filepath : str

            The checkpoint file path.

        """
        counters = self.get_counters()
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with lz4.frame.open(filepath, 'wb') as f:
            f.write(pickle.dumps(counters))

    def load_counters(self, filepath):
        r"""

        Restore the state of all internal counters.

        Parameters
        ----------
        filepath : str

            The checkpoint file path.

        """
        with lz4.frame.open(filepath, 'rb') as f:
            counters = pickle.loads(f.read())
        self.set_counters(counters)
