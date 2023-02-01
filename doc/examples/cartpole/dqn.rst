Cartpole with DQN
=================

In this notebook we solve the `CartPole <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment
using a simple :doc:`DQN </examples/stubs/dqn>` agent. Our function approximator is a multi-layer perceptron with one
hidden layer.

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
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/cartpole/dqn.ipynb

.. literalinclude:: dqn.py
