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

from .stochastic_q import StochasticQ


__all__ = (
    'StochasticRewardFunction',
)


class StochasticRewardFunction(StochasticQ):
    r"""

    A stochastic reward function :math:`p_\theta(r|s,a)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass.

    env : gym.Env

        The gym-style environment. This is used to validate the input/output structure of ``func``.

    value_range : tuple of floats, optional

        A pair of floats :code:`(min_value, max_value)`. If left unspecified, this defaults to
        :code:`value_range=env.reward_range`.

    observation_preprocessor : function, optional

        Turns a single observation into a batch of observations that in a form that is convenient
        for feeding into :code:`func`. If left unspecified, this defaults to:

        .. code:: python

            observation_preprocessor = default_preprocessor(env.observation_space)

        See :func:`coax.utils.default_preprocessor`.

    action_preprocessor : function, optional

        Turns a single action into a batch of actions that in a form that is convenient for feeding
        into :code:`func`. If left unspecified, this defaults to:

        .. code:: python

            action_preprocessor = default_preprocessor(env.action_space)

        See :func:`coax.utils.default_preprocessor`.

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
            self, func, env, value_range=None, num_bins=51, observation_preprocessor=None,
            action_preprocessor=None, value_transform=None, random_seed=None):

        super().__init__(
            func, env, value_range=(value_range or env.reward_range), num_bins=51,
            observation_preprocessor=None, action_preprocessor=None, value_transform=None,
            random_seed=None)
