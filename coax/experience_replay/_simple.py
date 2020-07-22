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

from collections import deque
from random import sample

import jax
import numpy as onp

from ..reward_tracing import TransitionBatch


__all__ = (
    'SimpleReplayBuffer',
)


class SimpleReplayBuffer:
    r"""
    A simple numpy implementation of an experience replay buffer. This is
    written primarily with computer game environments (Atari) in mind.

    It implements a generic experience replay buffer for environments in which
    individual observations (frames) are stacked to represent the state.

    Parameters
    ----------
    env : gym environment

        The main gym environment. This is needed to extract some metadata such
        as shapes and dtypes.

    capacity : positive int

        The capacity of the experience replay buffer. DQN typically uses
        ``capacity=1000000``.

    n : positive int, optional

        The number of steps over which to perform bootstrapping, i.e.
        :math:`n`-step bootstrapping.

    gamma : float between 0 and 1

        Reward discount factor.

    random_seed : int or None

        To get reproducible results.


    """
    def __init__(self, capacity, random_seed=None):
        self.capacity = int(capacity)
        self._rnd = onp.random.RandomState(random_seed)
        self.clear()

    def add(self, transition_batch):
        r"""
        Add a transition to the experience replay buffer.

        Parameters
        ----------
        transition_batch : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

        """
        self._deque.extend(transition_batch.to_singles())

    def sample(self, batch_size=32):
        r"""
        Get a batch of transitions to be used for bootstrapped updates.

        Parameters
        ----------
        batch_size : positive int, optional

            The desired batch size of the sample.

        Returns
        -------
        transitions : TransitionBatch

            A :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>` object.

        """
        def concatenate_leaves(pytrees):
            return jax.tree_multimap(lambda *leaves: onp.concatenate(leaves, axis=0), *pytrees)

        transitions = sample(self._deque, batch_size)
        return TransitionBatch(
            S=concatenate_leaves(t.S for t in transitions),
            A=concatenate_leaves(t.A for t in transitions),
            logP=concatenate_leaves(t.logP for t in transitions),
            Rn=concatenate_leaves(t.Rn for t in transitions),
            In=concatenate_leaves(t.In for t in transitions),
            S_next=concatenate_leaves(t.S_next for t in transitions),
            A_next=concatenate_leaves(t.A_next for t in transitions),
            logP_next=concatenate_leaves(t.logP_next for t in transitions),
        )

    def clear(self):
        r"""
        Clear the experience replay buffer.

        """
        self._deque = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self._deque)

    def __bool__(self):
        return bool(len(self))
