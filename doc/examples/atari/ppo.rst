Atari 2600: Pong with PPO
=========================

In this notebook we solve the `Pong <https://gymnasium.farama.org/environments/atari/pong/>`_ environment using a TD
actor-critic algorithm with :class:`PPO <coax.policy_objectives.PPOClip>` policy updates.

We use convolutional neural nets (without pooling) as our function approximator for the state value
function :math:`v(s)` and policy :math:`\pi(a|s)`.

In this version, the actor and critic don't share any weights. In other words, they each learn their
own feature extractor for the input state observations.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center

----

:download:`ppo.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/atari/ppo.ipynb

.. literalinclude:: ppo.py
