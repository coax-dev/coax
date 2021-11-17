r"""
Value Losses
============

.. autosummary::
    :nosignatures:

    coax.value_losses.mse
    coax.value_losses.huber
    coax.value_losses.logloss
    coax.value_losses.logloss_sign
    coax.value_losses.quantile_huber

----

This is a collection of loss functions that may be used for learning a value function. They are just
ordinary loss functions known from supervised learning.


Object Reference
----------------

.. autofunction:: coax.value_losses.mse
.. autofunction:: coax.value_losses.huber
.. autofunction:: coax.value_losses.logloss
.. autofunction:: coax.value_losses.logloss_sign
.. autofunction:: coax.value_losses.quantile_huber

"""

from ._losses import mse, huber, logloss, logloss_sign, quantile_huber


__all__ = (
    'mse',
    'huber',
    'logloss',
    'logloss_sign',
    'quantile_huber'
)
