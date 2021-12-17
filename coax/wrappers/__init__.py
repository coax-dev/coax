r"""
Wrappers
========

.. autosummary::
    :nosignatures:

    coax.wrappers.TrainMonitor
    coax.wrappers.FrameStacking
    coax.wrappers.BoxActionsToReals
    coax.wrappers.BoxActionsToDiscrete
    coax.wrappers.MetaPolicyEnv

----

OpenAI gym provides a nice modular interface to extend existing environments
using `environment wrappers
<https://github.com/openai/gym/tree/master/gym/wrappers>`_. Here we list some
wrappers that are used throughout the **coax** package.

The most notable wrapper that you'll probably want to use is
:class:`coax.wrappers.TrainMonitor`. It wraps the environment in a way that we
can view our training logs easily. It uses both the standard :py:mod:`logging`
module as well as tensorboard through the `tensorboardX
<https://tensorboardx.readthedocs.io/>`_ package.


Object Reference
----------------

.. autoclass:: coax.wrappers.TrainMonitor
.. autoclass:: coax.wrappers.FrameStacking
.. autoclass:: coax.wrappers.BoxActionsToReals
.. autoclass:: coax.wrappers.BoxActionsToDiscrete
.. autoclass:: coax.wrappers.MetaPolicyEnv


"""

from ._train_monitor import TrainMonitor
from ._frame_stacking import FrameStacking
from ._box_spaces import BoxActionsToReals, BoxActionsToDiscrete
from ._meta_policy import MetaPolicyEnv


__all__ = (
    'TrainMonitor',
    'FrameStacking',
    'BoxActionsToReals',
    'BoxActionsToDiscrete',
    'MetaPolicyEnv',
)
