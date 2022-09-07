""" From https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py """

import numpy as onp
from dm_control import suite
from dm_env import specs
from gym import envs, spaces, Env
from gym.envs.registration import register, make


class DMCWrapper(Env):

    def __init__(self, domain, task, seed):
        super().__init__()
        self.domain = domain
        self.task = task
        self.seed = seed
        self._make_env()

    def _make_env(self):
        self.dmc_env = suite.load(self.domain, self.task, task_kwargs=dict(random=self.seed))
        self.action_space = spec_to_box(self.dmc_env.action_spec(), dtype=onp.float32)
        self.observation_space = spec_to_box(
            *self.dmc_env.observation_spec().values(), dtype=onp.float64)

    def step(self, action):
        timestep = self.dmc_env.step(action)
        next_state, reward, terminated, truncated, info = flatten_obs(
            timestep.observation), timestep.reward, timestep.last(), False, {}
        return next_state, reward, terminated, truncated, info

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None and seed != self.seed:
            self._make_env()
        timestep = self.dmc_env.reset()
        return flatten_obs(timestep.observation)

    def render(self, mode="rgb_array"):
        return self.dmc_env.physics.render(
            height=84, width=84, camera_id=0
        )


def make_dmc(domain, task, seed=0):
    env_id = f"{domain}_{task}-v1"
    if env_id not in envs.registry:
        register(env_id, entry_point=DMCWrapper, kwargs=dict(
            domain=domain, task=task, seed=seed), max_episode_steps=1000)
    return make(env_id)


def extract_min_max(s):
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
    obs_pieces = []
    for v in obs.values():
        flat = onp.array([v]) if onp.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return onp.concatenate(obs_pieces, axis=0)
