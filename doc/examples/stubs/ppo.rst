Proximal Policy Optimization (PPO)
==================================

Consider the following the following importance-weighted off-policy objective:

.. math::
    :label: off_policy_objective

    J(\theta)\ =\ \mathbb{E}_t \left\{ \rho_\theta(S_t, A_t)\,\mathcal{A}(S_t, A_t) \right\}

where :math:`\mathcal{A}(s, a)` is the advantage function and :math:`\rho_\theta(s, a)` is the
probability ratio, defined as:

.. math::

    \rho_\theta(s, a)\ =\ \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{targ}}(a|s)}


The parameters :math:`\theta_\text{targ}` are the weights of the behavior policy, which is to say
:math:`A_t\sim\pi_{\theta_\text{targ}}(.|S_t)` in Eq. :eq:`off_policy_objective`.


**Importance sampling and outliers**

The use of the probability ratios like :math:`\rho_\theta(s, a)` is known as `importance sampling
<https://en.wikipedia.org/wiki/Importance_sampling>`_, which allows for us to create unbiased
estimates from out-of-distribution (off-policy) samples. A big problem with importance sampling is
that the probability ratios are unbounded from above, which often leads to *overestimation* or
*underestimation*.


**Mitigating overestimation and the PPO-clip objective**

The Proximal Policy Optimization (PPO) algorithm mitigates the problem of *overestimation*, leaving
underestimation uncorrected for. This mitigation is achieved by effectively clipping the probability
ratio in a specific way.

.. math::
    :label: ppo_clip_objective

    J(\theta; s,a)\ =\ \min\Big(
        \rho_\theta(s,a)\,\mathcal{A}(s,a)\,,\
        \bar{\rho}_\theta(s,a)\,\mathcal{A}(s,a)
    \Big)

where we introduced the clipped probability ratio:

.. math::

    \bar{\rho}_\theta(s,a)\ =\ \text{clip}(\rho_\theta(s,a), 1-\epsilon, 1+\epsilon)

The clipped estimate :math:`\bar{\rho}_\theta(s,a)\,\mathcal{A}(s,a)` removes both overestimation
and underestimation. Taking the minimal value between the unclipped and clipped estimates ensures
that we don't correct for *underestimation*. One reason to do this is that underestimation is
harmless, but a more important reason is that it provides a path towards higher values of the
expected advantage. In other words, not correcting for underestimation ensures that our objective
stays concave.


**Off-policy data collection**

A very nice property of the clipped surrogate objective it that it allows for slightly more
off-policy updates compared to the vanilla policy gradient. Moreover, it does this in a way that is
compatible with our ordinary first-order optimization techniques.

In other words, the PPO-clip objective allows for our behavior policy to differ slightly from the
current policy that's being updated. This makes more suitable for parallelization than the standard
REINFORCE-style policy objective, which is much more sensitive to off-policy deviations.


**Further reading**

This stub uses the same advantage actor-critic style setup as in :doc:`a2c`.

For more details on the PPO-clip objective, see the `PPO paper <https://arxiv.org/abs/1707.06347>`_.
For the **coax** implementation, have a look at :class:`coax.policy_objectives.PPOClip`.


----

:download:`ppo.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/stubs/ppo.ipynb

.. literalinclude:: ppo.py
