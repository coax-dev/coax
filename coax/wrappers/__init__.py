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
