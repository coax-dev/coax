Soft Q-Learning
===============

Soft q-learning is a variation of q-learning that it replaces the :code:`max` function by its soft
equivalent:

.. math::

    \text{max}^{(\tau)}_i x_i\ =\ \tau\,\log\sum_i \exp\left( x_i / \tau \right)

The temperature parameter :math:`\tau>0` determines the softness of the operation. We recover the
ordinary (hard) max function in the limit :math:`\tau\to0`.

The :math:`n`-step bootstrapped target is thus computed as

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,\tau\,\log\sum_a \exp\Bigl( q(S_{t+n}, a) / \tau \Bigr)

where

.. math::

    R^{(n)}_t\ =\ \sum_{k=0}^{n-1}\gamma^kR_{t+k}\ , \qquad
    I^{(n)}_t\ =\ \left\{\begin{matrix}
        0           & \text{if $S_{t+n}$ is a terminal state} \\
        \gamma^n    & \text{otherwise}
    \end{matrix}\right.

Soft q-learning (partially) mitigates over-estimation in the Bellman error. What's more, there is a
natural connection between soft q-learning and actor-critic methods (see `paper
<https://arxiv.org/abs/1704.06440>`_).


----

:download:`soft_qlearning.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/main/doc/_notebooks/stubs/soft_qlearning.ipynb

.. literalinclude:: soft_qlearning.py
