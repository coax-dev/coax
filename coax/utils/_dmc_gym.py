""" Adapted from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py """


import numpy as onp
from dm_control import suite
from dm_env import specs
from gymnasium import spaces, Env
from gymnasium.envs.registration import register, make, registry


def make_dmc(domain, task, seed=0, max_episode_steps=1000, height=84, width=84, camera_id=0):
    """
    Create a Gym environment for a DeepMind Control suite task.

    Parameters
    ----------
    domain : str
        Name of the domain.
    task : str
        Name of the task.
    seed : int
        Random seed.
    max_episode_steps : int
        Maximum number of steps per episode.
    height : int
        Height of the observation.
    width : int
        Width of the observation.
    camera_id : int
        Camera ID.

    Returns
    -------
    env : gymnasium.Env
        Gym environment.
    """
    env_id = f"{domain}_{task}-v1"
    if env_id not in registry:
        register(env_id, entry_point=DmcGymWrapper, kwargs=dict(
            domain=domain, task=task, seed=seed, height=height, width=width, camera_id=camera_id),
            max_episode_steps=max_episode_steps)
    return make(env_id)


class DmcGymWrapper(Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    r"""
    Wrapper for DeepMind Control suite environments.
    """

    def __init__(self, domain, task, seed, height, width, camera_id, render_mode='rgb_array'):
        super().__init__()
        self.domain = domain
        self.task = task
        self.seed = seed
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.render_mode = render_mode
        self._make_env()

    def _make_env(self):
        self.dmc_env = suite.load(self.domain, self.task, task_kwargs=dict(random=self.seed))
        self.action_space = spec_to_box(self.dmc_env.action_spec(), dtype=onp.float32)
        self.observation_space = spec_to_box(
            *self.dmc_env.observation_spec().values(), dtype=onp.float32)

    def step(self, action):
        timestep = self.dmc_env.step(action)
        next_state, reward, terminated, truncated, info = flatten_obs(
            timestep.observation), timestep.reward, timestep.last(), False, {}
        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
            self._make_env()
        timestep = self.dmc_env.reset()
        return flatten_obs(timestep.observation), {}

    def render(self):
        return self.dmc_env.physics.render(
            height=self.height, width=self.width, camera_id=self.camera_id
        )


def extract_min_max(s):
    r"""
    Extract min and max values from a dm_env.specs.ArraySpec.
    """
    assert s.dtype == onp.float64 or s.dtype == onp.float32
    dim = int(onp.prod(s.shape))
    if type(s) == specs.Array:
        bound = onp.inf * onp.ones(dim, dtype=onp.float32)
        return -bound, bound
    elif type(s) == specs.BoundedArray:
        zeros = onp.zeros(dim, dtype=onp.float32)
        return s.minimum + zeros, s.maximum + zeros
    else:
        raise ValueError("")


def spec_to_box(*spec, dtype):
    r"""
    Convert dm_env.specs.ArraySpec to gymnasium.spaces.Box.
    """
    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = onp.concatenate(mins, axis=0).astype(dtype)
    high = onp.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def flatten_obs(obs):
    r"""
    Flatten observation from a dictionary to a numpy array.
    """
    obs_pieces = []
    for v in obs.values():
        flat = onp.array([v]) if onp.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return onp.concatenate(obs_pieces, axis=0, dtype=onp.float32)
