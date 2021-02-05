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
Experience Replay
=================

.. autosummary::
    :nosignatures:

    coax.experience_replay.SimpleReplayBuffer
    coax.experience_replay.PrioritizedReplayBuffer

----

This is where we keep our experience-replay buffer classes. Replay buffers are typically used as
follows:

.. code:: python

    env = gym.make(...)

    # function approximator
    func = coax.FuncApprox(env)
    q = coax.Q(func)
    pi = coax.EpsilonGreedy(q, epsilon=0.1)

    # updater
    qlearning = coax.td_learning.QLearning(q)

    # reward tracer and replay buffer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
    buffer = coax.experience_replay.SimpleReplayBuffer(tracer, capacity=10000)


    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # trace n-step rewards and add to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            transition_batch = tracer.pop()  # batch_size = 1
            buffer.add(transition_batch)

        # sample random transitions from replay buffer
        transition_batch = buffer.sample(batch_size=32)
        qlearning.update(transition_batch)

        if done:
            break

        s = s_next



Object Reference
----------------

.. autoclass:: coax.experience_replay.SimpleReplayBuffer
.. autoclass:: coax.experience_replay.PrioritizedReplayBuffer


"""

from ._simple import SimpleReplayBuffer
from ._prioritized import PrioritizedReplayBuffer


__all__ = (
    'SimpleReplayBuffer',
    'PrioritizedReplayBuffer',
)
