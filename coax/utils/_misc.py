# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

import os
import time
import logging
import warnings
from collections import namedtuple
from importlib import reload, import_module
from types import ModuleType
from copy import deepcopy

import jax
import gym
import numpy as onp
from PIL import Image

from ..reward_tracing import TransitionSingle


__all__ = (
    'docstring',
    'enable_logging',
    'generate_gif',
    'get_env_attr',
    'get_transition',
    'has_env_attr',
    'is_policy',
    'is_qfunction',
    'is_vfunction',
    'reload_recursive',
    'render_episode',
    'getattr_safe',
    'StrippedEnv',
    'strip_env_recursive',
)


def docstring(obj):
    r'''

    A simple decorator that sets the ``__doc__`` attribute to ``obj.__doc__``
    on the decorated object, see example below.


    Parameters
    ----------
    obj : object

        The objects whose docstring you wish to copy onto the wrapped object.

    Examples
    --------
    >>> def f(x):
    ...     """Some docstring"""
    ...     return x * x
    ...
    >>> def g(x):
    ...     return 13 - x
    ...
    >>> g.__doc__ = f.__doc__

    This can abbreviated by:

    >>> @docstring(f)
    ... def g(x):
    ...     return 13 - x
    ...

    '''
    def decorator(func):
        func.__doc__ = obj.__doc__
        return func
    return decorator


def enable_logging(name=None, level=logging.INFO, output_filepath=None, output_level=None):
    r"""

    Enable logging output.

    This executes the following two lines of code:

    .. code:: python

        import logging
        logging.basicConfig(level=logging.INFO)


    Parameters
    ----------
    name : str, optional

        Name of the process that is logging. This can be set to whatever you
        like.

    level : int, optional

        Logging level for the default :py:class:`StreamHandler
        <logging.StreamHandler>`. The default setting is ``level=logging.INFO``
        (which is 20). If you'd like to see more verbose logging messages you
        might set ``level=logging.DEBUG``.

    output_filepath : str, optional

        If provided, a :py:class:`FileHandler <logging.FileHandler>` will be
        added to the root logger via:

        .. code:: python

            file_handler = logging.FileHandler(output_filepath)
            logging.getLogger('').addHandler(file_handler)

    output_level : int, optional

        Logging level for the :py:class:`FileHandler <logging.FileHandler>`. If
        left unspecified, this defaults to ``level``, i.e. the same level as
        the default :py:class:`StreamHandler <logging.StreamHandler>`.

    """
    if name is None:
        fmt = '[%(threadName)s|%(name)s|%(levelname)s] %(message)s'
    else:
        fmt = f'[{name}|%(threadName)s|%(name)s|%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=fmt)
    if output_filepath is not None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        fh = logging.FileHandler(output_filepath)
        fh.setLevel(level if output_level is None else output_level)
        logging.getLogger('').addHandler(fh)


def get_transition(env, random_seed=None):
    r"""
    Generate a single transition from the environment.

    This basically does a single step on the environment and then closes it.

    Parameters
    ----------
    env : gym environment

        A gym-style environment.

    random_seed : int, optional

        In order to generate the transition, we do some random sampling from the provided spaces.
        This `random_seed` set the seed for the pseudo-random number generators.

    Returns
    -------
    transition : TransitionSingle

        A single transition. Note that this can be turned into a
        :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` (with batchsize 1) by:

        .. code:: python

            transition_batch = transition.to_batch()

    """
    from ..wrappers import TrainMonitor
    if isinstance(env, TrainMonitor):
        env = env.env  # unwrap to strip off TrainMonitor

    s = env.reset()
    action_space = deepcopy(env.action_space)
    action_space.seed(random_seed)
    a = action_space.sample()
    a_next = action_space.sample()
    logp = None
    if isinstance(action_space, gym.spaces.Discrete):
        logp = -onp.log(action_space.n)
    if isinstance(action_space, gym.spaces.Box):
        sizes = action_space.high - action_space.low
        logp = -onp.sum(onp.log(sizes))  # log(prod(1/sizes))
    s_next, r, done, info = env.step(a)
    tr = TransitionSingle(
        s=s, a=a, logp=logp, r=r, done=done, info=info,
        s_next=s_next, a_next=a_next, logp_next=logp)
    return tr


def _reload(module, reload_all, reloaded, logger):
    if isinstance(module, ModuleType):
        module_name = module.__name__
    elif isinstance(module, str):
        module_name, module = module, import_module(module)
    else:
        raise TypeError(
            "'module' must be either a module or str; "
            f"got: {module.__class__.__name__}")

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        check = (
            # is it a module?
            isinstance(attr, ModuleType)

            # has it already been reloaded?
            and attr.__name__ not in reloaded

            # is it a proper submodule? (or just reload all)
            and (reload_all or attr.__name__.startswith(module_name))
        )
        if check:
            _reload(attr, reload_all, reloaded, logger)

    logger.debug(f"reloading module: {module_name}")
    reload(module)
    reloaded.add(module_name)


def reload_recursive(module, reload_external_modules=False):
    """
    Recursively reload a module (in order of dependence).

    Parameters
    ----------
    module : ModuleType or str

        The module to reload.

    reload_external_modules : bool, optional

        Whether to reload all referenced modules, including external ones which
        aren't submodules of ``module``.

    """
    logger = logging.getLogger('coax.utils.reload_recursive')
    _reload(module, reload_external_modules, set(), logger)


def render_episode(env, policy=None, step_delay_ms=0):
    r"""
    Run a single episode with env.render() calls with each time step.

    Parameters
    ----------
    env : gym environment

        A gym environment.

    policy : callable, optional

        A policy objects that is used to pick actions: ``a = policy(s)``. If left unspecified, we'll
        just take random actions instead, i.e. ``a = env.action_space.sample()``.

    step_delay_ms : non-negative float

        The number of milliseconds to wait between consecutive timesteps. This
        can be used to slow down the rendering.

    """
    from ..wrappers import TrainMonitor
    if isinstance(env, TrainMonitor):
        env = env.env  # unwrap to strip off TrainMonitor

    s = env.reset()
    env.render()

    for t in range(int(1e9)):
        a = env.action_space.sample() if policy is None else policy(s)
        s_next, r, done, info = env.step(a)

        env.render()
        time.sleep(step_delay_ms / 1e3)

        if done:
            break

        s = s_next

    time.sleep(5 * step_delay_ms / 1e3)


def has_env_attr(env, attr, max_depth=100):
    r"""
    Check if a potentially wrapped environment has a given attribute.

    Parameters
    ----------
    env : gym environment

        A potentially wrapped environment.

    attr : str

        The attribute name.

    max_depth : positive int, optional

        The maximum depth of wrappers to traverse.

    """
    e = env
    for i in range(max_depth):
        if hasattr(e, attr):
            return True
        if not hasattr(e, 'env'):
            break
        e = e.env

    return False


def get_env_attr(env, attr, default='__ERROR__', max_depth=100):
    r"""
    Get the given attribute from a potentially wrapped environment.

    Note that the wrapped envs are traversed from the outside in. Once the
    attribute is found, the search stops. This means that an inner wrapped env
    may carry the same (possibly conflicting) attribute. This situation is
    *not* resolved by this function.

    Parameters
    ----------
    env : gym environment

        A potentially wrapped environment.

    attr : str

        The attribute name.

    max_depth : positive int, optional

        The maximum depth of wrappers to traverse.

    """
    e = env
    for i in range(max_depth):
        if hasattr(e, attr):
            return getattr(e, attr)
        if not hasattr(e, 'env'):
            break
        e = e.env

    if default == '__ERROR__':
        raise AttributeError("env is missing attribute: {}".format(attr))

    return default


def generate_gif(
        env, policy, filepath,
        resize_to=None,
        duration=50,
        max_episode_steps=None):
    r"""
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment

        The environment to record from.

    policy : function s => a

        The policy that is used to take actions.

        .. code:: python

            a = policy(s)

        Therefore, ``policy`` just need to be a callable object that maps state observations to
        actions. For instance if ``pi`` is a :class:`coax.Policy`, we could execute a greedy policy
        by passing ``policy=pi.greedy``.

    filepath : str

        Location of the output gif file.

    resize_to : tuple of ints, optional

        The size of the output frames, ``(width, height)``. Notice the
        ordering: first **width**, then **height**. This is the convention PIL
        uses.

    duration : float, optional

        Time between frames in the animated gif, in milliseconds.

    max_episode_steps : int, optional

        The maximum number of step in the episode. If left unspecified, we'll
        attempt to get the value from ``env.spec.max_episode_steps`` and if
        that fails we default to 10000.

    """
    logger = logging.getLogger('generate_gif')
    max_episode_steps = max_episode_steps or \
        getattr(getattr(env, 'spec'), 'max_episode_steps', 10000)

    from ..wrappers import TrainMonitor
    if isinstance(env, TrainMonitor):
        env = env.env  # unwrap to strip off TrainMonitor

    # collect frames
    frames = []
    s = env.reset()
    for t in range(max_episode_steps):
        a = policy(s)
        s_next, r, done, info = env.step(a)

        # store frame
        frame = env.render(mode='rgb_array')
        frame = Image.fromarray(frame)
        frame = frame.convert('P', palette=Image.ADAPTIVE)
        if resize_to is not None:
            if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
                raise TypeError(
                    "expected a tuple of size 2, resize_to=(w, h)")
            frame = frame.resize(resize_to)

        frames.append(frame)

        if done:
            break

        s = s_next

    # store last frame
    frame = env.render(mode='rgb_array')
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
        frame = frame.resize(resize_to)
    frames.append(frame)

    # generate gif
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    frames[0].save(
        fp=filepath, format='GIF', append_images=frames[1:], save_all=True,
        duration=duration, loop=0)

    logger.info("recorded episode to: {}".format(filepath))


def is_vfunction(obj):
    r"""

    Check whether an object is a :class:`state value function <coax.V>`, or V-function.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a V-function.

    """
    # import at runtime to avoid circular dependence
    from .._core.value_v import V
    return isinstance(obj, V)


def is_qfunction(obj, qtype=None):
    r"""

    Check whether an object is a :class:`state-action value function <coax.Q>`, or Q-function.

    Parameters
    ----------
    obj

        Object to check.

    qtype : 1 or 2, optional

        Check for specific Q-function type, i.e. type-1 or type-2. See :class:`coax.Q` for more
        details.

    Returns
    -------
    bool

        Whether ``obj`` is a Q-function and (optionally) whether it specifically has qtype 1 or 2.

    """
    # import at runtime to avoid circular dependence
    from .._core.value_q import Q

    if qtype is None:
        return isinstance(obj, Q)
    if qtype not in (1, 2):
        raise ValueError("unexpected qtype: {}".format(qtype))
    return isinstance(obj, Q) and obj.qtype == qtype


def is_policy(obj, check_updateable=False):
    r"""

    Check whether an object is a :doc:`policy <policies>`.

    Parameters
    ----------
    obj

        Object to check.

    check_updateable : bool, optional

        If the obj is a policy, also check whether or not the policy is updateable.

    Returns
    -------
    bool

        Whether ``obj`` is an (updateable) policy.

    """
    # import at runtime to avoid circular dependence
    from .._base.abstract import BaseFunc, PolicyMixin, UpdateableMixin
    return (
        isinstance(obj, BaseFunc)
        and isinstance(obj, PolicyMixin)
        and (not check_updateable or isinstance(obj, UpdateableMixin)))


def getattr_safe(obj, name, default=None):
    """

    A safe implementation of :func:`getattr <python3:getattr>`. If an attr
    exists, but calling getattr raises an error, this implementation will
    silence the error and return the ``default`` value.

    Parameter
    ---------
    obj : object

        Any object.

    name : str

        The name of the attribute.

    default : object, optional

        The default value to return if getattr fails.

    Returns
    -------
    attr : object

        The attribute ``obj.name`` or ``default``.

    """
    attr = default
    try:
        attr = getattr(obj, name, default)
    except Exception:
        pass
    return attr


class StrippedEnv:
    r"""

    A version of an environment that both is static and respects the gym API.

    This is useful for creating pickleable environments that have a significantly reduced memory
    footprint.

    Parameters
    ----------
    env : gym environment

        The original (dynamic) gym-style environment.

    random_seed : int, optional

        The StrippedEnv creates a :class:`coax.TransitionSingle` to store its information
        internally. In order to generate this transition, we do some random sampling from the
        provided spaces. This `random_seed` set the seed for the pseudo-random number generators.

    """
    __slots__ = (
        'observation_space',
        'action_space',
        'reward_range',
        'spec',
        'metadata',
        # specific to StrippedEnv
        'transition',
        '_origname',
    )

    def __init__(self, env, random_seed=None):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range
        self.spec = env.spec
        self.metadata = env.metadata
        if hasattr(env, 'transition'):
            self.transition = env.transition
        else:
            self.transition = get_transition(env, random_seed=random_seed)
        self._origname = str(env)

    @classmethod
    def from_spaces(
            cls, observation_space, action_space,
            reward_range=None, spec=None, metadata=None, random_seed=None):
        r"""

        Create a new instance from ``observation_space`` and ``action_space``.

        Parameters
        ----------
        observation_space : gym-style space

            The space of state observations :math:`s`.

        action_space : gym-style space

            The space of actions :math:`a`.

        reward_range : pair of floats, optional

            The range of the rewards generated by the environment.

        spec : EnvSpec, optional

            The environment's EnvSpec. See the :mod:`gym.envs.register` module form more details.

        metadata : dict, optional

            The metadata dict of the environment.

        random_seed : int, optional

            The StrippedEnv creates a :class:`coax.TransitionSingle` to store its information
            internally. In order to generate this transition, we do some random sampling from the
            provided spaces. This `random_seed` set the seed for the pseudo-random number
            generators.

        """
        if reward_range is None:
            reward_range = (-float('inf'), float('inf'))
        if metadata is None:
            metadata = {}
        StaticEnv = namedtuple('StaticEnv', (
            'observation_space',
            'action_space',
            'reward_range',
            'spec',
            'metadata',
            'transition'))
        rnd = onp.random.RandomState(random_seed)
        observation_space.seed(random_seed)
        action_space.seed(random_seed)
        transition = TransitionSingle(
            s=jax.tree_map(onp.asanyarray, observation_space.sample()),
            a=jax.tree_map(onp.asanyarray, action_space.sample()),
            logp=onp.log(onp.clip(rnd.rand(), 1e-16, 1)),
            r=onp.clip(rnd.randn(), *reward_range),
            done=rnd.randint(2, dtype='bool'),
            info={},
            s_next=jax.tree_map(onp.asanyarray, observation_space.sample()),
            a_next=jax.tree_map(onp.asanyarray, action_space.sample()),
            logp_next=onp.log(onp.clip(rnd.rand(), 1e-16, 1)),
        )
        static_env = StaticEnv(
            observation_space=observation_space,
            action_space=action_space,
            reward_range=reward_range,
            spec=spec,
            metadata=metadata,
            transition=transition)
        return cls(static_env)

    def reset(self):
        return self.transition.s

    def step(self, a):
        t = self.transition
        return t.s_next, t.r, t.done, t.info

    def render(self, *args, **kwargs):
        raise NotImplementedError(
            "StrippedEnv is static; it cannot be rendered")

    def close(self):
        pass

    def seed(self, seed=None):
        warnings.warn("StrippedEnv is static; the seed will not be used")

    @property
    def unwrapped(self):
        return self

    def __str__(self):
        return f'<StrippedEnv<{self._origname}>>'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False


def strip_env_recursive(obj, depth=3):
    """

    Find all references to environments in an object and replace them by their stripped versions
    (using :class:`coax.utils.StrippedEnv`):

    .. code::

        env = StrippedEnv(env)

    Parameters
    ----------
    obj : FuncApprox or container object

        This could be either a :class:`FuncApprox` (or subclass thereof) or a container object that
        contains a :class:`FuncApprox` as one of its attributes.

    depth : non-negative int, optional

        How deep to recurse into the object to find an env to strip.

    """
    if depth < 0:
        return

    # import inline to avoid cyclic dependency
    from .._core.func_approx import FuncApprox

    if isinstance(obj, FuncApprox):
        if not isinstance(obj.env, StrippedEnv):
            obj.env = StrippedEnv(obj.env)
    else:
        for attr_name in dir(obj):
            if attr_name.startswith('__'):
                continue
            attr = getattr_safe(obj, attr_name)
            if attr is not None:
                strip_env_recursive(attr, depth - 1)

    return obj
