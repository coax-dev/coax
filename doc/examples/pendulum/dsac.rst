Pendulum with DSAC
==================

In this notebook we solve the `Pendulum <https://gymnasium.farama.org/environments/classic_control/pendulum/>`_ environment
using `DSAC`, the distributional variant of `SAC`. We follow the `implementation https://arxiv.org/abs/2004.14547>`
by using quantile regression to approximate the q function.

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
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/pendulum/dsac.ipynb

.. literalinclude:: dsac.py
