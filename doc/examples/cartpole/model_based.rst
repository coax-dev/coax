Cartpole with Model-Based agent
===============================

In this notebook we solve the `CartPole-v0 <https://gym.openai.com/envs/CartPole-v0/>`_ environment
using a model-based agent, which uses a function approximator for a value function :math:`v(s)` as
well as a dynamics model :math:`p(s'|s,a)`. Since the CartPole observation space covers the full
phase space of the dynamics, this agent is able to learn the task *within the first episode*.

If training is successful, this is what the result would look like:

.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


----

:download:`model_based.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/cartpole/model_based.ipynb

.. literalinclude:: model_based.py
