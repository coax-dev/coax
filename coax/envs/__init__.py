r"""
Environments
============

.. autosummary::
    :nosignatures:

    coax.envs.ConnectFourEnv

----

This is a collection of environments currently not included in
`Gymnasium <https://gymnasium.farama.org/>`_.


Object Reference
----------------

.. autoclass:: coax.envs.ConnectFourEnv

"""

from ._connect_four import ConnectFourEnv


__all__ = (
    'ConnectFourEnv',
)
