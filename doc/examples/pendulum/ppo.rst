Pendulum with PPO
=================

In this notebook we solve the `Pendulum <https://gym.openai.com/envs/Pendulum-v0/>`_ environment
using :doc:`PPO </examples/stubs/ppo>`. We'll use a simple multi-layer percentron for our function
approximator for the policy and q-function.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

.. note::

    Note that this agent is less robust than the DDPG agent. The reason is that we didn't do proper
    hyperparameter search for this agent. If you find a better set of hyperparameters, please let us
    know.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pendulum.gif
    :alt: Successfully swinging up the pendulum.
    :width: 360px
    :align: center

----

:download:`ppo.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/pendulum/ppo.ipynb

.. literalinclude:: ppo.py
