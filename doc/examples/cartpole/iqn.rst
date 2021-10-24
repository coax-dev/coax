Cartpole with IQN
=================

In this notebook we solve the `CartPole <https://gym.openai.com/envs/CartPole-v0/>`_ environment
using a simple :doc:`IQN </examples/stubs/iqn>` agent. Our function approximator is an implicit quantile network that
approximates the quantiles of the state-action value function.

We chose not to use an experience-replay buffer, which makes training a little volatile. Feel free
to add a replay buffer if you want to make the training more robust.

If training is successful, this is what the result would look like:

.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


----

:download:`dqn.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/cartpole/iqn.ipynb

.. literalinclude:: dqn.py
