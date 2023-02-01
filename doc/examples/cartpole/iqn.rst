Cartpole with IQN
====================

In this notebook we solve the `CartPole <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment
using a simple :doc:`IQN </examples/stubs/iqn>` agent. Our function approximator is an Implicit Quantile Network that
approximates the quantiles of the state-action value function.

If training is successful, this is what the result would look like:

.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


----

:download:`iqn.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/cartpole/iqn.ipynb

.. literalinclude:: iqn.py
