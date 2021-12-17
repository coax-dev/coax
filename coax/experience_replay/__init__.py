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
