Implicit Quantile Network (IQN)
===============================

Implicit Quantile Networks are a distributional RL method that model the distribution of returns 
using quantile regression. They were introduced in the paper
[`arxiv:1806.06923 <https://arxiv.org/abs/1806.06923>`_] and replaced the fixed parametrization of the quantile q-function
of Quantile-Regression DQN [`arxiv:1710.10044 <https://arxiv.org/abs/1710.10044>`_] with uniformly sampled quantile fractions.

For the generation of equally spaced quantile fractions as in QR-DQN in **coax** have a look 
at :class:`coax.utils.quantiles`. For uniformly distributed quantile fractions as in IQN there
is the :class:`coax.utils.quantiles_uniform` function.


----

:download:`iqn.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/stubs/iqn.ipynb

.. literalinclude:: iqn.py
