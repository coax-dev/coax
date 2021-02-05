Atari 2600: Pong with Ape-X DQN
===============================

In this notebook we solve the `Pong <https://gym.openai.com/envs/Pong-v0/>`_ environment using a
distrbuted agent known as `Ape-X DQN <https://arxiv.org/abs/1803.00933>`. This agent has multiple rollout workers (actors), one learner and a parameter server.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`apex_dqn.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/main/doc/_notebooks/atari/apex_dqn.ipynb

.. literalinclude:: apex_dqn.py
