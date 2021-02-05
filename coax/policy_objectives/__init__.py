# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

r"""
Policy Objectives
=================

.. autosummary::
    :nosignatures:

    coax.policy_objectives.VanillaPG
    coax.policy_objectives.PPOClip
    coax.policy_objectives.DeterministicPG


----

This is a collection of policy objectives that can be used in policy-gradient
methods.


Object Reference
----------------

.. autoclass:: coax.policy_objectives.VanillaPG
.. autoclass:: coax.policy_objectives.PPOClip
.. autoclass:: coax.policy_objectives.DeterministicPG

"""

from ._base import PolicyObjective
from ._vanilla_pg import VanillaPG
from ._ppo_clip import PPOClip
from ._deterministic_pg import DeterministicPG


__all__ = (
    'PolicyObjective',
    'VanillaPG',
    # 'CrossEntropy',  # TODO
    'PPOClip',
    'DeterministicPG',
)
