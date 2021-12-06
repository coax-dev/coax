Pendulum with SAC
==================

In this notebook we solve the `Pendulum <https://gym.openai.com/envs/Pendulum-v0/>`_ environment
using `SAC`. We'll use a simple multi-layer percentron for our function
approximator for the policy and q-function.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pendulum.gif
    :alt: Successfully swinging up the pendulum.
    :width: 360px
    :align: center

----

:download:`sac.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/pendulum/sac.ipynb

.. literalinclude:: sac.py
