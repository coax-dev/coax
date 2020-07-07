FrozenLake with Mode-Based Q-Learning
=====================================

In this notebook we solve a non-slippery version of the `FrozenLake-v0
<https://gym.openai.com/envs/FrozenLake-v0/>`_ environment using value-based
control with q-learning bootstrap targets.

We'll use a linear function approximator for our state-action value function
:math:`q_\theta(s,a)`. Since the observation space is discrete, this is
equivalent to the table-lookup case.

This version differs from :doc:`./qlearning` in that it uses mode-based
:class:`QLearningMode <coax.td_learning.QLearningMode>` updater class instead
of the default :class:`QLearning <coax.td_learning.QLearning>` (which is
specific to discrete actions).


.. image:: /_static/img/frozen_lake_q.gif
    :alt: Non-Slippery FrozenLake solved
    :width: 132px
    :align: center


----

:download:`qlearning_mode.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/frozen_lake/qlearning_mode.ipynb

.. literalinclude:: qlearning_mode.py
