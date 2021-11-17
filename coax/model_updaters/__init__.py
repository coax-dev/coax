r"""
Model Updaters
==============

.. autosummary::
    :nosignatures:

    coax.model_updaters.ModelUpdater

----

This is a collection of objects that are used to update dynamics models, i.e. transition models and
reward functions.


Object Reference
----------------

.. autoclass:: coax.model_updaters.ModelUpdater

"""

# TODO(krholshe): think of better names for this module and classes

from ._model_updater import ModelUpdater


__all__ = (
    'ModelUpdater',
)
