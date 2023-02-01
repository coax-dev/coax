Atari 2600: Pong with DQN & Prioritized Experience Replay
=========================================================

In this notebook we solve the `Pong <https://gymnasium.farama.org/environments/atari/pong/>`_ environment using a
version of a :doc:`DQN </examples/stubs/dqn>` agent, trained using a :class:`PrioritizedReplayBuffer
<coax.experience_replay.PrioritizedReplayBuffer>` instead of the standard :class:`SimpleReplayBuffer
<coax.experience_replay.SimpleReplayBuffer>`.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`dqn_per.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/atari/dqn_per.ipynb

.. literalinclude:: dqn_per.py
