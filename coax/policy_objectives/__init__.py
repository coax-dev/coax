r"""
Policy Objectives
=================

.. autosummary::
    :nosignatures:

    coax.policy_objectives.VanillaPG
    coax.policy_objectives.PPOClip
    coax.policy_objectives.DeterministicPG
    coax.policy_objectives.SoftPG


----

This is a collection of policy objectives that can be used in policy-gradient
methods.


Object Reference
----------------

.. autoclass:: coax.policy_objectives.VanillaPG
.. autoclass:: coax.policy_objectives.PPOClip
.. autoclass:: coax.policy_objectives.DeterministicPG
.. autoclass:: coax.policy_objectives.SoftPG

"""

from ._base import PolicyObjective
from ._vanilla_pg import VanillaPG
from ._ppo_clip import PPOClip
from ._deterministic_pg import DeterministicPG
from ._soft_pg import SoftPG


__all__ = (
    'PolicyObjective',
    'VanillaPG',
    # 'CrossEntropy',  # TODO
    'PPOClip',
    'DeterministicPG',
    'SoftPG'
)
