FrozenLake with A2C
===================

In this notebook we solve a non-slippery version of the `FrozenLake-v0
<https://gymnasium.farama.org/environments/toy_text/frozen_lake/>`_ environment using the TD actor critic algorithm with
vanilla policy gradient updates. This algorithm is known as the Advantage Actor Critic, abbreviated
A2C.

We'll use a linear function approximator for our policy :math:`\pi_\theta(a|s)`
and our state value function :math:`v_\theta(s)`. Since the observation space
is discrete, this is equivalent to the table-lookup case.


.. image:: /_static/img/frozen_lake_pi.gif
    :alt: Non-Slippery FrozenLake solved
    :width: 132px
    :align: center


----

:download:`a2c.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/frozen_lake/a2c.ipynb

.. literalinclude:: a2c.py
