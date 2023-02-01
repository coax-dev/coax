Atari 2600: Pong with DQN (with Soft Q-Learning)
================================================

In this notebook we solve the `Pong <https://gymnasium.farama.org/environments/atari/pong/>`_ environment using the
classic :doc:`DQN </examples/stubs/dqn>` agent. We'll use a convolutional neural net (without
pooling) as our function approximator for the q-function.

This notebook is different from the regular implementation in that the q-function is updated using
:doc:`soft q-learning </examples/stubs/soft_qlearning>` instead of :doc:`ordinary q-learning
</examples/stubs/qlearning>`.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`dqn_soft.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/atari/dqn_soft.ipynb

.. literalinclude:: dqn_soft.py
