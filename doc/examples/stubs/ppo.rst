Proximal Policy Optimization (PPO)
==================================

The main advantage of using PPO over the standard REINFORCE-style policy objective is that it allows
for our behavior policy to differ slightly from the current policy that's being updated. This makes
more suitable for parallelization.

This stub uses the same advantage actor-critic style setup as in :doc:`a2c`.

For more details on the PPO-clip objective, see the PPO paper [`arxiv:1707.06347
<https://arxiv.org/abs/1707.06347>`_]. For the **coax** implementation, have a look at
:class:`coax.policy_objectives.PPOClip`.


----

:download:`ppo.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/stubs/ppo.ipynb

.. literalinclude:: ppo.py
