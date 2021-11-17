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
    coax.td_learning.SoftClippedDoubleQLearning

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
.. autoclass:: coax.td_learning.SoftClippedDoubleQLearning


"""

from ._simple_td import SimpleTD
from ._sarsa import Sarsa
from ._expectedsarsa import ExpectedSarsa
from ._qlearning import QLearning
from ._doubleqlearning import DoubleQLearning
from ._softqlearning import SoftQLearning
from ._clippeddoubleqlearning import ClippedDoubleQLearning
from ._softclippeddoubleqlearning import SoftClippedDoubleQLearning


__all__ = (
    'SimpleTD',
    'Sarsa',
    'ExpectedSarsa',
    'QLearning',
    'DoubleQLearning',
    'SoftQLearning',
    'ClippedDoubleQLearning',
    'SoftClippedDoubleQLearning'
)
