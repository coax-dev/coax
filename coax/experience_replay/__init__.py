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
Experience Replay
=================

.. autosummary::
    :nosignatures:

    coax.experience_replay.SimpleReplayBuffer
    coax.experience_replay.PrioritizedReplayBuffer

----

This is where we keep our experience-replay buffer classes. Some examples of agents that use a
replay buffer are:

- :doc:`/examples/stubs/dqn`
- :doc:`/examples/stubs/dqn_per`.

For specific examples, have a look at the :doc:`agents for Atari games </examples/atari/index>`.


Object Reference
----------------

.. autoclass:: coax.experience_replay.SimpleReplayBuffer
.. autoclass:: coax.experience_replay.PrioritizedReplayBuffer


"""

from ._simple import SimpleReplayBuffer
from ._prioritized import PrioritizedReplayBuffer


__all__ = (
    'SimpleReplayBuffer',
    'PrioritizedReplayBuffer',
)
