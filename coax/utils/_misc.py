import os
import time
import logging
from importlib import reload, import_module
from types import ModuleType

import jax.numpy as jnp
import numpy as onp
import pandas as pd
import lz4.frame
import cloudpickle as pickle
from PIL import Image


__all__ = (
    'docstring',
    'enable_logging',
    'dump',
    'dumps',
    'load',
    'loads',
    'generate_gif',
    'get_env_attr',
    'getattr_safe',
    'has_env_attr',
    'is_policy',
    'is_qfunction',
    'is_reward_function',
    'is_stochastic',
    'is_transition_model',
    'is_vfunction',
    'pretty_repr',
    'pretty_print',
    'reload_recursive',
    'render_episode',
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
        fmt = '[%(name)s|%(levelname)s] %(message)s'
    else:
        fmt = f'[{name}|%(name)s|%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=fmt)
    if output_filepath is not None:
        os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)
        fh = logging.FileHandler(output_filepath)
        fh.setLevel(level if output_level is None else output_level)
        logging.getLogger('').addHandler(fh)


def dump(obj, filepath):
    r"""

    Save an object to disk.

    Parameters
    ----------
    obj : object

        Any python object.

    filepath : str

        Where to store the instance.

    Warning
    -------

    References between objects are only preserved if they are stored as part of a single object, for
    example:

    .. code:: python

        # b has a reference to a
        a = [13]
        b = {'a': a}

        # references preserved
        dump((a, b), 'ab.pkl.lz4')
        a_new, b_new = load('ab.pkl.lz4')
        b_new['a'].append(7)
        print(b_new)  # {'a': [13, 7]}
        print(a_new)  # [13, 7]         <-- updated

        # references not preserved
        dump(a, 'a.pkl.lz4')
        dump(b, 'b.pkl.lz4')
        a_new = load('a.pkl.lz4')
        b_new = load('b.pkl.lz4')
        b_new['a'].append(7)
        print(b_new)  # {'a': [13, 7]}
        print(a_new)  # [13]            <-- not updated!!

    Therefore, the safest way to create checkpoints is to store the entire state as a single object
    like a dict or a tuple.

    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with lz4.frame.open(filepath, 'wb') as f:
        f.write(pickle.dumps(obj))


def dumps(obj):
    r"""

    Serialize an object to an lz4-compressed pickle byte-string.

    Parameters
    ----------
    obj : object

        Any python object.

    Returns
    -------
    s : bytes

        An lz4-compressed pickle byte-string.

    Warning
    -------

    References between objects are only preserved if they are stored as part of a single object, for
    example:

    .. code:: python

        # b has a reference to a
        a = [13]
        b = {'a': a}

        # references preserved
        s = dumps((a, b))
        a_new, b_new = loads(s)
        b_new['a'].append(7)
        print(b_new)  # {'a': [13, 7]}
        print(a_new)  # [13, 7]         <-- updated

        # references not preserved
        s_a = dumps(a)
        s_b = dumps(b)
        a_new = loads(s_a)
        b_new = loads(s_b)
        b_new['a'].append(7)
        print(b_new)  # {'a': [13, 7]}
        print(a_new)  # [13]            <-- not updated!!

    Therefore, the safest way to create checkpoints is to store the entire state as a single object
    like a dict or a tuple.

    """
    return lz4.frame.compress(pickle.dumps(obj))


def load(filepath):
    r"""

    Load an object from a file that was created by :func:`dump(obj, filepath) <dump>`.

    Parameters
    ----------
    filepath : str

        File to load.

    """
    with lz4.frame.open(filepath, 'rb') as f:
        return pickle.loads(f.read())


def loads(s):
    r"""

    Load an object from a byte-string that was created by :func:`dumps(obj) <dumps>`.

    Parameters
    ----------
    s : str

        An lz4-compressed pickle byte-string.

    """
    return pickle.loads(lz4.frame.decompress(s))


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

        The number of milliseconds to wait between consecutive timesteps. This can be used to slow
        down the rendering.

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


def generate_gif(env, filepath, policy=None, resize_to=None, duration=50, max_episode_steps=None):
    r"""
    Store a gif from the episode frames.

    Parameters
    ----------
    env : gym environment

        The environment to record from.

    filepath : str

        Location of the output gif file.

    policy : callable, optional

        A policy objects that is used to pick actions: ``a = policy(s)``. If left unspecified, we'll
        just take random actions instead, i.e. ``a = env.action_space.sample()``.

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
    max_episode_steps = max_episode_steps \
        or getattr(getattr(env, 'spec'), 'max_episode_steps', 10000)

    from ..wrappers import TrainMonitor
    if isinstance(env, TrainMonitor):
        env = env.env  # unwrap to strip off TrainMonitor

    # collect frames
    frames = []
    s = env.reset()
    for t in range(max_episode_steps):
        a = env.action_space.sample() if policy is None else policy(s)
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
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    frames[0].save(
        fp=filepath, format='GIF', append_images=frames[1:], save_all=True,
        duration=duration, loop=0)

    logger.info("recorded episode to: {}".format(filepath))


def is_transition_model(obj):
    r"""

    Check whether an object is a dynamics model.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a dynamics function.

    """
    # import at runtime to avoid circular dependence
    from .._core.transition_model import TransitionModel
    from .._core.stochastic_transition_model import StochasticTransitionModel
    return isinstance(obj, (TransitionModel, StochasticTransitionModel))


def is_reward_function(obj):
    r"""

    Check whether an object is a dynamics model.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a dynamics function.

    """
    # import at runtime to avoid circular dependence
    from .._core.reward_function import RewardFunction
    from .._core.stochastic_reward_function import StochasticRewardFunction
    return isinstance(obj, (RewardFunction, StochasticRewardFunction))


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
    from .._core.v import V
    from .._core.stochastic_v import StochasticV
    return isinstance(obj, (V, StochasticV))


def is_qfunction(obj):
    r"""

    Check whether an object is a :class:`state-action value function <coax.Q>`, or Q-function.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a Q-function and (optionally) whether it is of modeltype 1 or 2.

    """
    # import at runtime to avoid circular dependence
    from .._core.q import Q
    from .._core.stochastic_q import StochasticQ
    from .._core.successor_state_q import SuccessorStateQ
    return isinstance(obj, (Q, StochasticQ, SuccessorStateQ))


def is_stochastic(obj):
    r"""

    Check whether an object is a stochastic function approximator.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a stochastic function approximator.

    """
    # import at runtime to avoid circular dependence
    from .._core.policy import Policy
    from .._core.stochastic_v import StochasticV
    from .._core.stochastic_q import StochasticQ
    from .._core.stochastic_reward_function import StochasticRewardFunction
    from .._core.stochastic_transition_model import StochasticTransitionModel
    return isinstance(obj, (
        Policy, StochasticV, StochasticQ, StochasticRewardFunction,
        StochasticTransitionModel))


def is_policy(obj):
    r"""

    Check whether an object is a :doc:`policy <policies>`.

    Parameters
    ----------
    obj

        Object to check.

    Returns
    -------
    bool

        Whether ``obj`` is a policy.

    """
    # import at runtime to avoid circular dependence
    from .._core.policy import Policy
    from .._core.value_based_policy import EpsilonGreedy, BoltzmannPolicy
    return isinstance(obj, (Policy, EpsilonGreedy, BoltzmannPolicy))


def pretty_repr(o, d=0):
    r"""

    Generate pretty :func:`repr` (string representions).

    Parameters
    ----------
    o : object

        Any object.

    d : int, optional

        The depth of the recursion. This is used to determine the indentation level in recursive
        calls, so we typically keep this 0.

    Returns
    -------
    pretty_repr : str

        A nicely formatted string representation of :code:`object`.

    """
    i = "  "  # indentation string
    if isinstance(o, (jnp.ndarray, onp.ndarray, pd.Index)):
        try:
            summary = f", min={onp.min(o):.3g}, median={onp.median(o):.3g}, max={onp.max(o):.3g}"
        except Exception:
            summary = ""
        return f"array(shape={o.shape}, dtype={str(o.dtype)}{summary:s})"
    if isinstance(o, (pd.Series, pd.DataFrame)):
        sep = ',\n' + i * (d + 1)
        items = zip(('index', 'data'), (o.index, o.values))
        body = sep + sep.join(f"{k}={pretty_repr(v, d + 1)}" for k, v in items)
        return f"{type(o).__name__}({body})"
    if hasattr(o, '_asdict'):
        sep = '\n' + i * (d + 1)
        body = sep + sep.join(f"{k}={pretty_repr(v, d + 1)}" for k, v in o._asdict().items())
        return f"{type(o).__name__}({body})"
    if isinstance(o, tuple):
        sep = ',\n' + i * (d + 1)
        body = '\n' + i * (d + 1) + sep.join(f"{pretty_repr(v, d + 1)}" for v in o)
        return f"({body})"
    if isinstance(o, list):
        sep = ',\n' + i * (d + 1)
        body = '\n' + i * (d + 1) + sep.join(f"{pretty_repr(v, d + 1)}" for v in o)
        return f"[{body}]"

    if hasattr(o, 'items'):
        sep = ',\n' + i * (d + 1)
        body = '\n' + i * (d + 1) + sep.join(
            f"{repr(k)}: {pretty_repr(v, d + 1)}" for k, v in o.items())
        return f"{{{body}}}"
    return repr(o)


def pretty_print(obj):
    r"""

    Print :func:`pretty_repr(obj) <coax.utils.pretty_repr>`.

    Parameters
    ----------
    obj : object

        Any object.

    """
    print(pretty_repr(obj))


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
