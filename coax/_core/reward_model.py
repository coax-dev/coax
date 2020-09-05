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

import numpy as onp
from gym.spaces import Box

from ..proba_dists import ProbaDist
from .base_model import BaseModel


__all__ = (
    'RewardModel',
)


class RewardModel(BaseModel):
    r"""

    A parametrized reward-function, represented by a stochastic function :math:`p_\theta(r|s,a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    observation_space : gym.Space

        The observation space of the environment. This is used to generate example input for
        initializing :attr:`params`.

    action_space : gym.Space

        The action space of the environment. This is used to generate example input for
        initializing :attr:`params`.

    reward_range : tuple of floats

        A pair of floats :code:`(min_reward, max_reward)`, which is typically provided by the
        environment as :code:`env.reward_range`.

    observation_preprocessor

        Turns a single observation into a batch of observations that are compatible with the
        corresponding probability distribution. If left unspecified, this defaults to:

        .. code:: python

            observation_preprocessor = ProbaDist(observation_space).preprocess_variate

        See also :attr:`coax.proba_dists.ProbaDist.preprocess_variate`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions that are compatible with the corresponding
        probability distribution. If left unspecified, this defaults to:

        .. code:: python

            action_preprocessor = ProbaDist(action_space).preprocess_variate

        See also :attr:`coax.proba_dists.ProbaDist.preprocess_variate`.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, observation_space, action_space, reward_range,
            observation_preprocessor=None, action_preprocessor=None, random_seed=None):

        self.reward_range = reward_range
        if observation_preprocessor is None:
            observation_preprocessor = ProbaDist(observation_space).preprocess_variate
        if action_preprocessor is None:
            action_preprocessor = ProbaDist(action_space).preprocess_variate
        proba_dist = self._reward_proba_dist(reward_range)

        super().__init__(
            func=func,
            observation_space=observation_space,
            action_space=action_space,
            observation_preprocessor=observation_preprocessor,
            action_preprocessor=action_preprocessor,
            proba_dist=proba_dist,
            random_seed=random_seed)

    @classmethod
    def example_data(
            cls, observation_space, action_space, reward_range, observation_preprocessor=None,
            action_preprocessor=None, batch_size=1, random_seed=None):

        if observation_preprocessor is None:
            observation_preprocessor = ProbaDist(observation_space).preprocess_variate
        if action_preprocessor is None:
            action_preprocessor = ProbaDist(action_space).preprocess_variate
        proba_dist = cls._reward_proba_dist(reward_range)

        return super().example_data(
            observation_space=observation_space,
            action_space=action_space,
            observation_preprocessor=observation_preprocessor,
            action_preprocessor=action_preprocessor,
            proba_dist=proba_dist,
            batch_size=batch_size,
            random_seed=random_seed)

    def __call__(self, s, a=None, return_logp=False):
        r"""

        Sample a reward :math:`r` from the reward function :math:`p(r|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        return_logp : bool, optional

            Whether to return the log-propensity :math:`\log p(s'|s,a)`.

        Returns
        -------
        r : float or list thereof

            Depending on whether :code:`a` is provided, this either returns a single reward
            :math:`r` or a list of :math:`n` rewards, one for each discrete action.

        logp : non-positive float or list thereof, optional

            The log-propensity :math:`\log p(r|s,a)`. This is only returned if we set
            ``return_logp=True``. Depending on whether :code:`a` is provided, this is either a
            single float or a list of :math:`n` floats, one for each discrete action.

        """
        return super().__call__(s, a=a, return_logp=return_logp)

    def mode(self, s, a=None):
        r"""

        Get the most probable successor state :math:`r` according to the reward model,
        :math:`r=\arg\max_{r}p_\theta(r|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        r : float or list thereof

            Depending on whether :code:`a` is provided, this either returns a single reward
            :math:`r` or a list of :math:`n` rewards, one for each discrete action.

        """
        return super().mode(s, a=a)

    def dist_params(self, s, a=None):
        r"""

        Get the parameters of the conditional probability distribution :math:`p_\theta(r|s,a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        dist_params : dict or list of dicts

            Depending on whether :code:`a` is provided, this either returns a single dist-params
            dict or a list of :math:`n` such dicts, one for each discrete action.

        """
        return super().dist_params(s, a=a)

    @staticmethod
    def _reward_proba_dist(reward_range):
        assert len(reward_range) == 2
        low, high = onp.clip(reward_range[0], -1e6, 1e6), onp.clip(reward_range[1], -1e6, 1e6)
        assert low < high, f"inconsistent low, high: {low}, {high}"
        reward_space = Box(low, high, shape=())
        return ProbaDist(reward_space)
