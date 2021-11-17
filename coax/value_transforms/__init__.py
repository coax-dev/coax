r"""
Value Transforms
================

.. autosummary::
    :nosignatures:

    coax.value_transforms.ValueTransform
    coax.value_transforms.LogTransform

----

This module contains some useful **value transforms**. These are functions
that can be used to rescale or warp the returns for more a more robust training
signal, see e.g. :class:`coax.value_transforms.LogTransform`.


Object Reference
----------------

.. autoclass:: coax.value_transforms.ValueTransform
.. autoclass:: coax.value_transforms.LogTransform


"""
from ._base import ValueTransform
from ._log_transform import LogTransform


__all__ = (
    'ValueTransform',
    'LogTransform',
)
