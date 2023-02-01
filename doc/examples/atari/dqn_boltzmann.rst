Atari 2600: Pong with DQN (with Boltzmann policy)
=================================================

In this notebook we solve the `Pong <https://gymnasium.farama.org/environments/atari/pong/>`_ environment using the
classic :doc:`DQN </examples/stubs/dqn>` agent. We'll use a convolutional neural net (without
pooling) as our function approximator for the q-function.

This notebook is different from the regular implementation in that it uses a **Boltzmann policy**
instead of an epsilon-greedy policy to ensure sufficient exploration.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`dqn_boltzmann.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/atari/dqn_boltzmann.ipynb

.. literalinclude:: dqn_boltzmann.py
