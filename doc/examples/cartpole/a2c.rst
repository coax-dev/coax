Cartpole with Advantage Actor-Critic (A2C)
==========================================

In this notebook we solve the `CartPole-v0 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment
using a simple TD actor-critic, also known as an advantage actor-critic (A2C). Our function
approximator is a simple multi-layer perceptron with one hidden layer.

If training is successful, this is what the result would look like:

.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


----

:download:`a2c.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/cartpole/a2c.ipynb

.. literalinclude:: a2c.py
