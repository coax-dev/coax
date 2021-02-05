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
Reward Tracing
==============

.. autosummary::
    :nosignatures:

    coax.reward_tracing.NStep
    coax.reward_tracing.MonteCarlo
    coax.reward_tracing.TransitionBatch

----

The term **reward tracing** refers to the process of turning raw experience into
:class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` objects. These
:class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` objects are then used to learn, i.e.
to update our function approximators.

Reward tracing typically entails keeping some episodic cache in order to relate a state :math:`S_t`
or state-action pair :math:`(S_t, A_t)` to a collection of objects that can be used to construct a
target (feedback signal):

.. math::

    \left(R^{(n)}_t, I^{(n)}_t, S_{t+n}, A_{t+n}\right)

where

.. math::

    R^{(n)}_t\ &=\ \sum_{k=0}^{n-1}\gamma^kR_{t+k} \\
    I^{(n)}_t\ &=\ \left\{\begin{matrix}
        0           & \text{if $S_{t+n}$ is a terminal state} \\
        \gamma^n    & \text{otherwise}
    \end{matrix}\right.

For example, in :math:`n`-step SARSA target is constructed as:

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,q(S_{t+n}, A_{t+n})



Object Reference
----------------

.. autoclass:: coax.reward_tracing.NStep
.. autoclass:: coax.reward_tracing.MonteCarlo
.. autoclass:: coax.reward_tracing.TransitionBatch

"""

from ._transition import TransitionBatch
from ._montecarlo import MonteCarlo
from ._nstep import NStep

__all__ = (
    'TransitionBatch',
    'MonteCarlo',
    'NStep',
)
