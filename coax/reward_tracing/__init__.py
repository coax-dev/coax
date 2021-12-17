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
