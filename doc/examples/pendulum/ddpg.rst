Pendulum with DDPG
==================

In this notebook we solve the `Pendulum-v0 <https://gym.openai.com/envs/Pendulum-v0/>`_ environment
using :doc:`DDPG </examples/stubs/ddpg>`. We'll use a simple multi-layer percentron for our function
approximator for the policy and q-function.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pendulum.gif
    :alt: Successfully swinging up the pendulum.
    :align: center

----

:download:`ddpg.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/pendulum/ddpg.ipynb

.. literalinclude:: ddpg.py
