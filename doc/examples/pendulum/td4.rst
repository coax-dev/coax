Pendulum with TD4
==================

In this notebook we solve the `Pendulum <https://gymnasium.farama.org/environments/classic_control/pendulum/>`_ environment
using TD4 which is the distributional variant of :doc:`TD3 </examples/stubs/td3>`. We estimate the q function using quantile
regression as in :doc:`IQN </examples/stubs/iqn>`.

This notebook periodically generates GIFs, so that we can inspect how the training is progressing.

After a few hundred episodes, this is what you can expect:

.. image:: /_static/img/pendulum.gif
    :alt: Successfully swinging up the pendulum.
    :width: 360px
    :align: center

----

:download:`td4.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/pendulum/td4.ipynb

.. literalinclude:: td4.py
