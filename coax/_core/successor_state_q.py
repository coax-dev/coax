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
from gym.spaces import Discrete

from ..utils import safe_sample
from .._core.value_v import V


__all__ = (
    'SuccessorStateQ',
)


class SuccessorStateQ:
    r"""

    A state-action value function :math:`q(s,a)=r(s,a)+\gamma\mathop{\mathbb{E}}_{s'\sim
    p(.|s,a)}v(s')`.

    Parameters
    ----------
    v : V

        A state value function :math:`v(s)`.

    p : DynamicsModel

        A dynamics model :math:`p(s'|s,a)`. This may also be a ordinary function with the signature:
        :code:`(Observation, Action) -> Observation`.

    r : RewardFunction

        A reward function :math:`r(s,a)`. This may also be a ordinary function with the signature:
        :code:`(Observation, Action) -> float`.

    gamma : float between 0 and 1, optional

        The discount factor for future rewards :math:`\gamma\in[0,1]`.

    random_seed : int, optional

        Seed for pseudo-random number generators.

    """
    def __init__(self, v, p, r, gamma=0.9):
        self._check_functions(v, p, r)
        self.v = v
        self.p = p
        self.r = r
        self.gamma = gamma

    @staticmethod
    def _check_functions(v, p, r):
        if not isinstance(v, V):
            raise TypeError(f"v must be of type V, got: {type(v)}")
        s = safe_sample(v.observation_space)
        a = safe_sample(v.action_space)
        try:
            assert p(s, a) in v.observation_space, "s_next is not an element of observation_space"
        except Exception as e:
            raise TypeError(
                "the dynamics model p(s'|s,a) generated a invalid successor state; "
                f"caught exception: {e}")
        try:
            float(r(s, a))
        except Exception as e:
            raise TypeError(
                f"the reward model r(s,a) generated an invalid reward; caught exception: {e}")

    def __call__(self, s, a=None):
        r"""

        Evaluate the state-action function on a state observation :math:`s` or
        on a state-action pair :math:`(s, a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action

            A single action :math:`a`.

        Returns
        -------
        q_sa or q_s : ndarray

            Depending on whether :code:`a` is provided, this either returns a scalar representing
            :math:`q(s,a)\in\mathbb{R}` or a vector representing :math:`q(s,.)\in\mathbb{R}^n`,
            where :math:`n` is the number of discrete actions. Naturally, this only applies for
            discrete action spaces.

        """
        v, p, r, γ = self.v, self.p, self.r, self.gamma
        if a is None:
            if not isinstance(self.v.action_space, Discrete):
                raise TypeError(
                    "input 'a' is required for q-function when action space is non-Discrete")
            return onp.stack([r(s, a) + γ * v(p(s, a)) for a in range(self.v.action_space.n)])
        return r(s, a) + γ * v(p(s, a))
