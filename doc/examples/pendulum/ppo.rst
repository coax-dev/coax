Pendulum with PPO
=================

In this notebook we solve the `Pendulum-v0 <https://gym.openai.com/envs/Pendulum-v0/>`_ environment
using a TD actor-critic algorithm with :class:`PPO <coax.policy_objectives.PPOClip>` policy updates.

We use a simple multi-layer percentron as our function approximators for the state value function
:math:`v(s)` and policy :math:`\pi(a|s)`.

This algorithm is slow to converge (if it does at all). You should start to see improvement in the
average return after about 150k timesteps. Below you'll see a particularly succesful episode:


.. image:: /_static/img/pendulum.gif
    :alt: A particularly succesful episode of Pendulum.
    :align: center


----

:download:`ppo.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/pendulum/ppo.ipynb

.. literalinclude:: ppo.py
