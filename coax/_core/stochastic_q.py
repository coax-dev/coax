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

from gym.spaces import Box

from ..utils import default_preprocessor
from ..proba_dists import DiscretizedIntervalDist
from ..value_transforms import ValueTransform
from .base_stochastic_func_type1 import BaseStochasticFuncType1


__all__ = (
    'StochasticQ',
)


class StochasticQ(BaseStochasticFuncType1):
    r"""

    A q-function :math:`q(s,a)`, represented by a stochastic function
    :math:`\mathbb{P}_\theta(G_t|S_t=s,A_t=a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    value_range : tuple of floats

        A pair of floats :code:`(min_value, max_value)`.

    num_bins : int, optional

        The space of rewards is discretized in :code:`num_bins` equal sized bins. We use the default
        setting of 51 as suggested in the `Distributional RL <https://arxiv.org/abs/1707.06887>`_
        paper.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations in a form that is convenient for
        feeding into :code:`func`. If left unspecified, this defaults to
        :func:`default_preprocessor(env.observation_space) <coax.utils.default_preprocessor>`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions in a form that is convenient for feeding into
        :code:`func`. If left unspecified, this defaults
        :func:`default_preprocessor(env.action_space) <coax.utils.default_preprocessor>`.

    value_transform : ValueTransform or pair of funcs, optional

        If provided, the target for the underlying function approximator is transformed:

        .. math::

            \tilde{G}_t\ =\ f(G_t)

        This means that calling the function involves undoing this transformation using its inverse
        :math:`f^{-1}`. The functions :math:`f` and :math:`f^{-1}` are given by
        ``value_transform.transform_func`` and ``value_transform.inverse_func``, respectively. Note
        that a ValueTransform is just a glorified pair of functions, i.e. passing
        ``value_transform=(func, inverse_func)`` works just as well.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(
            self, func, env, value_range, num_bins=51, observation_preprocessor=None,
            action_preprocessor=None, value_transform=None, random_seed=None):

        self.value_transform = value_transform
        self.value_range = self._check_value_range(value_range)
        proba_dist = self._get_proba_dist(self.value_range, value_transform, num_bins)

        # set defaults
        if observation_preprocessor is None:
            observation_preprocessor = default_preprocessor(env.observation_space)
        if action_preprocessor is None:
            action_preprocessor = default_preprocessor(env.action_space)
        if self.value_transform is None:
            self.value_transform = ValueTransform(lambda x: x, lambda x: x)
        if not isinstance(self.value_transform, ValueTransform):
            self.value_transform = ValueTransform(*value_transform)

        super().__init__(
            func=func,
            observation_space=env.observation_space,
            action_space=env.action_space,
            observation_preprocessor=observation_preprocessor,
            action_preprocessor=action_preprocessor,
            proba_dist=proba_dist,
            random_seed=random_seed)

    @property
    def num_bins(self):
        return self.proba_dist.space.n

    @classmethod
    def example_data(
            cls, env, value_range, num_bins=51, observation_preprocessor=None,
            action_preprocessor=None, value_transform=None, batch_size=1, random_seed=None):

        value_range = cls._check_value_range(value_range)
        proba_dist = cls._get_proba_dist(value_range, value_transform, num_bins)

        if observation_preprocessor is None:
            observation_preprocessor = default_preprocessor(env.observation_space)
        if action_preprocessor is None:
            action_preprocessor = default_preprocessor(env.action_space)

        return super().example_data(
            env=env,
            observation_preprocessor=observation_preprocessor,
            action_preprocessor=action_preprocessor,
            proba_dist=proba_dist,
            batch_size=batch_size,
            random_seed=random_seed)

    def __call__(self, s, a=None, return_logp=False):
        r"""

        Sample a value.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        return_logp : bool, optional

            Whether to return the log-propensity associated with the sampled output value.

        Returns
        -------
        value : float or list thereof

            Depending on whether :code:`a` is provided, this either returns a single value or a list
            of :math:`n` values, one for each discrete action.

        logp : non-positive float or list thereof, optional

            The log-propensity associated with the sampled output value. This is only returned if we
            set ``return_logp=True``. Depending on whether :code:`a` is provided, this is either a
            single float or a list of :math:`n` floats, one for each discrete action.

        """
        return super().__call__(s, a=a, return_logp=return_logp)

    def mean(self, s, a=None):
        r"""

        Get the mean value.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        value : float or list thereof

            Depending on whether :code:`a` is provided, this either returns a single value or a list
            of :math:`n` values, one for each discrete action.

        """
        return super().mean(s, a=a)

    def mode(self, s, a=None):
        r"""

        Get the most probable value.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action, optional

            A single action :math:`a`. This is *required* if the actions space is non-discrete.

        Returns
        -------
        value : float or list thereof

            Depending on whether :code:`a` is provided, this either returns a single value or a list
            of :math:`n` values, one for each discrete action.

        """
        return super().mode(s, a=a)

    def dist_params(self, s, a=None):
        r"""

        Get the parameters of the underlying (conditional) probability distribution.

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
    def _get_proba_dist(value_range, value_transform, num_bins):
        if value_transform is not None:
            f, _ = value_transform
            value_range = f(value_range[0]), f(value_range[1])
        reward_space = Box(*value_range, shape=())
        return DiscretizedIntervalDist(reward_space, num_bins)

    @staticmethod
    def _check_value_range(value_range):
        if not (isinstance(value_range, (tuple, list))
                and len(value_range) == 2
                and isinstance(value_range[0], (int, float))
                and isinstance(value_range[1], (int, float))
                and value_range[0] < value_range[1]):
            raise TypeError("value_range is not a valid pair tuple of floats: (low, high)")
        return float(value_range[0]), float(value_range[1])
