Deep Q-Network (DQN)
====================

Deep Q-Network (DQN) is somewhat of a misnomer. It came about the seminal DQN
paper [`arxiv:1312.5602 <1312.5602>`_], which used a deep neural net as the
function approximator for the q-function. DQN has since come to mean:
*q-learning with experience replay and a target network*.

For the **coax** implementation of q-learning and experience replay, have a
look at :class:`coax.td_learning.QLearning` and
:class:`coax.ExperienceReplayBuffer`. The *target network* is just a copy of
the main q-function. Note that the target network does need to be synchonized
every once in a while. This is done by periodically applying
exponential-smoothing updates.


----

:download:`dqn.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/stubs/dqn.ipynb

.. literalinclude:: dqn.py
