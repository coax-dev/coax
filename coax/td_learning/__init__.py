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

r"""
TD Learning
===========

.. autosummary::
    :nosignatures:

    coax.td_learning.SimpleTD
    coax.td_learning.Sarsa
    coax.td_learning.ExpectedSarsa
    coax.td_learning.QLearning
    coax.td_learning.DoubleQLearning
    coax.td_learning.SoftQLearning
    coax.td_learning.ClippedDoubleQLearning

----

This is a collection of objects that are used to update value functions via *Temporal Difference*
(TD) learning. A state value function :class:`coax.V` is using :class:`coax.td_learning.SimpleTD`.
To update a state-action value function :class:`coax.Q`, there are multiple options available. The
difference between the options are the manner in which the TD-target is constructed.


Object Reference
----------------

.. autoclass:: coax.td_learning.SimpleTD
.. autoclass:: coax.td_learning.Sarsa
.. autoclass:: coax.td_learning.ExpectedSarsa
.. autoclass:: coax.td_learning.QLearning
.. autoclass:: coax.td_learning.DoubleQLearning
.. autoclass:: coax.td_learning.SoftQLearning
.. autoclass:: coax.td_learning.ClippedDoubleQLearning


"""

from ._simple_td import SimpleTD
from ._sarsa import Sarsa
from ._expectedsarsa import ExpectedSarsa
from ._qlearning import QLearning
from ._doubleqlearning import DoubleQLearning
from ._softqlearning import SoftQLearning
from ._clippeddoubleqlearning import ClippedDoubleQLearning


__all__ = (
    'SimpleTD',
    'Sarsa',
    'ExpectedSarsa',
    'QLearning',
    'DoubleQLearning',
    'SoftQLearning',
    'ClippedDoubleQLearning',
)
