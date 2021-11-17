import inspect

import gym


class MetaPolicyEnv(gym.Wrapper):
    r"""

    Wrap a gym-style environment such that it may be used by a meta-policy,
    i.e. a bandit that selects a policy (an *arm*), which is then used to
    sample a lower-level action and fed the original environment. In other
    words, the actions that the :attr:`step` method expects are *meta-actions*,
    selecting different *arms*. The lower-level actions (and their
    log-propensities) that are sampled internally are stored in the ``info``
    dict, returned by the :attr:`step` method.

    Parameters
    ----------
    env : gym-style environment

        The original environment to be wrapped into a meta-policy env.

    \*arms : functions

        Callable objects that take a state observation :math:`s` and return an
        action :math:`a` (and optionally, log-propensity :math:`\log\pi(a|s)`).
        See for example :attr:`coax.Policy.__call__` or
        :attr:`coax.Policy.mode`.

    """
    def __init__(self, env, *arms):
        super().__init__(env)
        self.arms = arms
        self.action_space = gym.spaces.Discrete(len(arms))
        self._s = None

    def reset(self):
        self._s = self.env.reset()
        return self._s

    def step(self, a_meta):
        assert self.action_space.contains(a_meta), "a_meta is invalid"
        assert self._s is not None, "please call env.reset() first"

        pi = self.arms[a_meta]
        if 'return_logp' in inspect.getargspec(pi).args:
            a, logp = pi(self._s, return_logp=True)
        else:
            a, logp = pi(self._s), 0.

        self._s, r, done, info = self.env.step(a)
        info = info or {}
        info.update({'a': a, 'logp': logp})
        return self._s, r, done, info
