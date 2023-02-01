Cartpole with Model-Based agent
===============================

In this notebook we solve the `CartPole-v0 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ environment
using a model-based agent, which uses a function approximator for a value function :math:`v(s)` as
well as a dynamics model :math:`p(s'|s,a)`. Since the CartPole observation space covers the full
phase space of the dynamics, this agent is able to learn the task *within the first episode*.

The way in which the dynamics model is used in this agent is rather simple. Namely, we only use it
to define a single-step look-ahead q-function, i.e.

.. math::

    q(s,a)\ =\ r(s,a) + \mathop{\mathbb{E}}_{s'\sim p_\theta(.|s,a)} v_\theta(s')

This composite q-function is implemented by :class:`coax.SuccessorStateQ`. Note that the reward
function for the CartPole environment is simply :math:`r(s,a)=1` at each time step, so we don't need
to model that.


If training is successful, this is what the result would look like:

.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


----

:download:`model_based.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/cartpole/model_based.ipynb

.. literalinclude:: model_based.py
