Atari 2600: Pong with DQN (mode-based q-learning)
=================================================

In this notebook we solve the `PongDeterministic-v4 <https://gym.openai.com/envs/Pong-v0/>`_
environment using the classic :doc:`DQN </examples/stubs/dqn>` agent. We'll use a convolutional
neural net (without pooling) as our function approximator for the q-function.

This version differs from the standard DQN version in that it uses mode-based :class:`QLearningMode
<coax.td_learning.QLearningMode>` updater class instead of the default :class:`QLearning
<coax.td_learning.QLearning>` (which is specific to discrete actions).

Also, :class:`QLearningMode <coax.td_learning.QLearningMode>` expects a **type-I** q-function (see
:doc:`this example <dqn_type1>`).

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`dqn_mode.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/atari/dqn_mode.ipynb

.. literalinclude:: dqn_mode.py
