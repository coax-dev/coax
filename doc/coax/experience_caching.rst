Experience Caching
==================

In RL we often make use of data caching. This might be short-term caching, over
the course of an episode, or it might be long-term caching as is done in
experience replay.


Episodic Cache
--------------

These short-term caching objects allow us to cache experience within an
episode. For instance :class:`MonteCarloCache <coax.caching.MonteCarloCache>`
caches all transitions collected over an entire episode and then gives us back
the the :math:`\gamma`-discounted returns when the episode finishes.

Another short-term caching object is :class:`NStepCache
<coax.caching.NStepCache>`, which keeps an :math:`n`-sized sliding window
of transitions that allows us to do :math:`n`-step bootstrapping.


Experience Replay Buffer
------------------------

At the moment, we only have one long-term caching object, which is the
:class:`ExperienceReplayBuffer <coax.caching.ExperienceReplayBuffer>`.
This object can hold an arbitrary number of transitions; the only constraint is
the amount of available memory on your machine.

The way we use learn from the experience stored in the
:class:`ExperienceReplayBuffer <coax.caching.ExperienceReplayBuffer>` is
by sampling from it and then feeding the batch of transitions to our function approximator.


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.MonteCarloCache
    coax.NStepCache
    coax.ExperienceReplayBuffer


.. autoclass:: coax.MonteCarloCache
.. autoclass:: coax.NStepCache
.. autoclass:: coax.ExperienceReplayBuffer

