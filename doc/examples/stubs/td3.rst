Twin-Delayed DDPG (TD3)
=======================

The `TD3 <https://arxiv.org/abs/1802.09477>`_ algorithm is a variant of :doc:`DDPG <ddpg>`, which
replaces the ordinary q-learning updates by *double* q-learning updates i, in which the
:math:`n`-step bootstrapped target is constructed as:

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,\min_{i,j}q_i(S_{t+n}, \arg\max_a q_j(S_{t+n}, a))

The rest of the agent is essentially the same as that of :doc:`DDPG <ddpg>`.

----

:download:`td3.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/stubs/td3.ipynb

.. literalinclude:: td3.py
