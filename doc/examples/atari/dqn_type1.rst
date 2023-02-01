Atari 2600: Pong with DQN (type-I q-function)
=============================================

In this notebook we solve the `Pong <https://gymnasium.farama.org/environments/atari/pong/>`_ environment using the
classic :doc:`DQN </examples/stubs/dqn>` agent. We'll use a convolutional neural net (without
pooling) as our function approximator for the q-function.

This version differs from the standard DQN version in that the q-function is modelled as a **type-I
q-function**, i.e.

.. math::

    (s, a)\ \mapsto\ q(s,a) \in \mathbb{R}

instead of the standard (type-II):

.. math::

    s\ \mapsto\ q(s, .) \in \mathbb{R}^n

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`dqn_type1.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/atari/dqn_type1.ipynb

.. literalinclude:: dqn_type1.py
