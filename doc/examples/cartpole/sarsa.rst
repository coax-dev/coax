Cartpole with SARSA
===================

In this notebook we solve the `CartPole-v0
<https://gym.openai.com/envs/CartPole-v0/>`_ environment using a simple SARSA
agent. Our function approximator is a multi-layer perceptron with one hidden
layer.

If training is successful, this is what the result would look like:

.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


----

:download:`sarsa.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/cartpole/sarsa.ipynb

.. literalinclude:: sarsa.py
