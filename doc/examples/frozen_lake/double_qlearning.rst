FrozenLake with Double Q-Learning
=================================

In this notebook we solve a non-slippery version of the `FrozenLake-v0
<https://gymnasium.farama.org/environments/toy_text/frozen_lake/>`_ environment using value-based control with double
q-learning bootstrap targets.

We'll use a linear function approximator for our state-action value function :math:`q_\theta(s,a)`.
Since the observation space is discrete, this is equivalent to the table-lookup case.


.. image:: /_static/img/frozen_lake_q.gif
    :alt: Non-Slippery FrozenLake solved
    :width: 132px
    :align: center


----

:download:`double_qlearning.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/frozen_lake/double_qlearning.ipynb

.. literalinclude:: double_qlearning.py
