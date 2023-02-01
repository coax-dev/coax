import gymnasium
import jax.numpy as jnp
import numpy as onp

from ..utils import docstring
from .policy import Policy


__all__ = (
    'RandomPolicy',
)


class RandomPolicy:
    r"""

    A simple random policy.

    Parameters
    ----------
    env : gymnasium.Env

        The gymnasium-style environment. This is only used to get the :code:`env.action_space`.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """
    def __init__(self, env, random_seed=None):
        if not isinstance(env.action_space, gymnasium.Space):
            raise TypeError(
                f"env.action_space must be a gymnasium.Space, got: {type(env.action_space)}")
        self.action_space = env.action_space
        self.action_space.seed(random_seed)
        self.random_seed = random_seed

    @docstring(Policy.__call__)
    def __call__(self, s, return_logp=False):
        a = self.action_space.sample()
        if not return_logp:
            return a

        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            logp = -onp.log(self.num_actions)
            return a, logp

        if isinstance(self.action_space, gymnasium.spaces.Box):
            sizes = self.action_space.high - self.action_space.low
            logp = -onp.sum(onp.log(sizes))  # log(prod(1/sizes))
            return a, logp

        raise NotImplementedError(
            "the log-propensity of a 'uniform' distribution over a "
            f"{self.action_space.__class__.__name__} space is not yet implemented; "
            "please submit a feature request")

    @docstring(Policy.mode)
    def mode(self, s):
        return self(s, return_logp=False)

    @docstring(Policy.dist_params)
    def dist_params(self, s):
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            return {'logits': jnp.zeros(self.action_space.n)}

        if isinstance(self.action_space, gymnasium.spaces.Box):
            return {
                'mu': jnp.zeros(self.action_space.shape),
                'logvar': 15 * jnp.ones(self.action_space.shape)}

        raise NotImplementedError(
            "the dist_params of a 'uniform' distribution over a "
            f"{self.action_space.__class__.__name__} space is not yet implemented; "
            "please submit a feature request")
