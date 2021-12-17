r"""
Regularizers
============

.. autosummary::
    :nosignatures:

    coax.regularizers.EntropyRegularizer
    coax.regularizers.KLDivRegularizer

----

This is a collection of regularizers that can be used to put soft constraints on stochastic function
approximators. These is typically added to the loss/objective to avoid premature exploitation of a
policy.


Object Reference
----------------

.. autoclass:: coax.regularizers.EntropyRegularizer
.. autoclass:: coax.regularizers.KLDivRegularizer

"""

from ._entropy import Regularizer
from ._entropy import EntropyRegularizer
from ._kl_div import KLDivRegularizer
from ._nstep_entropy import NStepEntropyRegularizer


__all__ = (
    'Regularizer',
    'EntropyRegularizer',
    'KLDivRegularizer',
    'NStepEntropyRegularizer'
)
