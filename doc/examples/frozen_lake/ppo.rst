FrozenLake with PPO
===================

In this notebook we solve a non-slippery version of the `FrozenLake-v0
<https://gymnasium.farama.org/environments/toy_text/frozen_lake/>`_ environment using the TD actor critic algorithm with
:class:`PPO <coax.policy_objectives.PPOClip>` policy updates.

We'll use a linear function approximator for our policy :math:`\pi_\theta(a|s)` and our state value
function :math:`v_\theta(s)`. Since the observation space is discrete, this is equivalent to the
table-lookup case.


.. image:: /_static/img/frozen_lake_pi.gif
    :alt: Non-Slippery FrozenLake solved
    :width: 132px
    :align: center


----

:download:`ppo.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/frozen_lake/ppo.ipynb

.. literalinclude:: ppo.py
