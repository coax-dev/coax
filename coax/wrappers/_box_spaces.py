import gym
import numpy as onp
from scipy.special import expit as sigmoid

from .._base.mixins import AddOrigToInfoDictMixin


__all__ = (
    'BoxActionsToReals',
    'BoxActionsToDiscrete',
)


class BoxActionsToReals(gym.Wrapper, AddOrigToInfoDictMixin):
    r"""

    This wrapper decompactifies a :class:`Box <gym.spaces.Box>` action space to the reals. This is
    required in order to be able to use a Gaussian policy.

    In practice, the wrapped environment expects the input action
    :math:`a_\text{real}\in\mathbb{R}^n` and then it compactifies it back to a Box of the right
    size:

    .. math::

        a_\text{box}\ =\ \text{low} + (\text{high}-\text{low})\times\text{sigmoid}(a_\text{real})

    Technically, the transformed space is still a Box, but that's only because we assume that the
    values lie between large but finite bounds, :math:`a_\text{real}\in[-10^{15}, 10^{15}]^n`.

    """
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise NotImplementedError("BoxActionsToReals is only implemented for Box action spaces")

        shape_flat = onp.prod(self.env.action_space.shape),
        self.action_space = gym.spaces.Box(
            low=onp.full(shape_flat, -1e15, self.env.action_space.dtype),
            high=onp.full(shape_flat, 1e15, self.env.action_space.dtype))

    def step(self, a):
        assert self.action_space.contains(a)
        self._a_orig = self._compactify(a)
        s_next, r, done, info = super().step(self._a_orig)
        self._add_a_orig_to_info_dict(info)
        return s_next, r, done, info

    def _compactify(self, action):
        hi, lo = self.env.action_space.high, self.env.action_space.low
        action = onp.clip(action, -1e15, 1e15)
        action = onp.reshape(action, self.env.action_space.shape)
        return lo + (hi - lo) * sigmoid(action)


class BoxActionsToDiscrete(gym.Wrapper, AddOrigToInfoDictMixin):
    r"""

    This wrapper splits a :class:`Box <gym.spaces.Box>` action space into bins. The resulting action
    space is either :class:`Discrete <gym.spaces.Discrete>` or :class:`MultiDiscrete
    <gym.spaces.MultiDiscrete>`, depending on the shape of the original action space.

    Parameters
    ----------
    num_bins : int or tuple of ints

        The number of bins to use. A multi-dimenionsional box requires a tuple of num_bins instead
        of a single integer.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, env, num_bins, random_seed=None):
        super().__init__(env)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise NotImplementedError(
                "BoxActionsToDiscrete is only implemented for Box action spaces")
        self._rnd = onp.random.RandomState(random_seed)
        self._init_action_space(num_bins)  # also sets self._nvec and self._size

    def step(self, a):
        assert self.action_space.contains(a)
        self._a_orig = self._discrete_to_box(a)
        s_next, r, done, info = super().step(self._a_orig)
        self._add_a_orig_to_info_dict(info)
        return s_next, r, done, info

    def _discrete_to_box(self, a_discrete):
        hi, lo = self.env.action_space.high, self.env.action_space.low
        a_flat = (a_discrete + self._rnd.rand(self._size)) / self._nvec
        a_reshaped = onp.reshape(a_flat, self.env.action_space.shape)
        a_rescaled = lo + a_reshaped * (hi - lo)
        return a_rescaled

    def _init_action_space(self, num_bins):
        self._size = onp.prod(self.env.action_space.shape)
        if isinstance(num_bins, int):
            self._nvec = [num_bins] * self._size
        elif isinstance(num_bins, tuple) and all(isinstance(i, int) for i in num_bins):
            if len(num_bins) != self._size:
                raise ValueError(
                    "len(num_bins) must be equal to the number of non-trivial dimensions: "
                    f"{self._size}")
            self._nvec = onp.asarray(num_bins)
        else:
            raise TypeError("num_bins must an int or tuple of ints")

        if self._size == 1:
            self.action_space = gym.spaces.Discrete(self._nvec[0])
        else:
            self.action_space = gym.spaces.MultiDiscrete(self._nvec)
