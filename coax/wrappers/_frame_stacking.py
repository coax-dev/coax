from collections import deque

import gymnasium


class FrameStacking(gymnasium.Wrapper):
    r"""

    Wrapper that does frame stacking (see `DQN paper
    <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_).

    This implementation is different from most implementations in that it doesn't perform the
    stacking itself. Instead, it just returns a tuple of frames (untouched), which may be stacked
    downstream.

    The benefit of this implementation is two-fold. First, it respects the :mod:`gymnasium.spaces`
    API, where each observation is truly an element of the observation space (this is not true of
    the gymnasium implementation, which uses a custom data class to maintain its minimal memory
    footprint). Second, this implementation is compatibility with the :mod:`jax.tree_util` module,
    which means that we can feed it into jit-compiled functions directly.

    Example
    -------

    .. code::

        import gymnasium
        env = gymnasium.make('PongNoFrameskip-v0')
        print(env.observation_space)  # Box(210, 160, 3)

        env = FrameStacking(env, num_frames=2)
        print(env.observation_space)  # Tuple((Box(210, 160, 3), Box(210, 160, 3)))

    Parameters
    ----------
    env : gymnasium-style environment

        The original environment to be wrapped.

    num_frames : positive int

        Number of frames to stack.

    """
    def __init__(self, env, num_frames):
        if not (isinstance(num_frames, int) and num_frames > 0):
            raise TypeError(f"num_frames must be a positive int, got: {num_frames}")

        super().__init__(env)
        self.observation_space = gymnasium.spaces.Tuple((self.env.observation_space,) * num_frames)
        self._frames = deque(maxlen=num_frames)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self._frames.append(observation)
        return tuple(self._frames), reward, done, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._frames.extend(observation for _ in range(self._frames.maxlen))
        return tuple(self._frames), info  # shallow copy
