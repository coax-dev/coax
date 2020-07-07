Deep Deterministic Policy Gradients (DDPG)
==========================================

The Deep Deterministic Policy Gradients (DDPG) algorithm is a little different
from other policy objectives. It learns a policy directly from a (type-I)
q-function. The

.. math::

    J(\theta; s,a)\ =\ q_\varphi(s, a_\theta(s))

Here :math:`a_\theta(s)` is the *mode* of the underlying conditional
probability distribution :math:`\pi_\theta(.|s)`. See e.g. the :attr:`mode`
method of :class:`coax.proba_dists.NormalDist`. In other words, we evaluate the
policy according to the current estimate of its best-case performance. This is
implemented by the :class:`coax.policy_objectives.DeterministicPG` updater
class.

Since the policy objective uses a q-function :math:`q_\varphi(s,a)`, we also
need to learn that. At the moment of writing, there are two ways to learn
:math:`q_\varphi(s,a)` in **coax**.

**Option 1: SARSA.**

The first option is to use SARSA updates,
whose :math:`n`-step bootstrapped target is constructed as:

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,
        q_{\varphi_\text{targ}}\!(S_{t+n}, A_{t+n})

where :math:`A_{t+n}` is *sampled from experience* and

.. math::

    R^{(n)}_t\ &=\ \sum_{k=0}^{n-1}\gamma^kR_{t+k} \\
    I^{(n)}_t\ &=\ \left\{\begin{matrix}
        0           & \text{if $S_{t+n}$ is a terminal state} \\
        \gamma^n    & \text{otherwise}
    \end{matrix}\right.

This is implemented by the :class:`coax.td_learning.Sarsa` updater class.

**Option 2: Q-Learning.**

The second option is to use q-learning updates, whose :math:`n`-step
bootstrapped target is instead constructed as:

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,q_{\varphi_\text{targ}}\!\left(
        S_{t+n}, a_{\theta_\text{targ}}\!(s)\right)

Here, :math:`a_{\theta_\text{targ}}\!(s)` is the mode introduced above,
evaluated on the target-model weights :math:`\theta_\text{targ}`. The reason
why we call this q-learning is that we construct the TD-target as though the
next action :math:`A_{t+n}` would have been the greedy action. This is
implemented by the :class:`coax.td_learning.QLearningMode` updater class.

For more details, have a look at the **spinningup** page on DDPG :doc:`here
<spinup:algorithms/ddpg>`, which includes links to the original papers.

----

:download:`ddpg.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/stubs/ddpg.ipynb

.. literalinclude:: ddpg.py
