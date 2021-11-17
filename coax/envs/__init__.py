r"""
Environments
============

.. autosummary::
    :nosignatures:

    coax.envs.ConnectFourEnv

----

This is a collection of environments currently not included in `OpenAI Gym
<https://gym.openai.com/>`_.


Object Reference
----------------

.. autoclass:: coax.envs.ConnectFourEnv

"""

from ._connect_four import ConnectFourEnv


__all__ = (
    'ConnectFourEnv',
)
