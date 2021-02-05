Workers
=======

.. autosummary::
    :nosignatures:

    coax.Worker

----

This module provides the abstractions required for building distributed agents.

The :class:`coax.Worker` is at the heart of such agents.

The way this works in **coax** is to define a class derived from :class:`coax.Worker` and then to
create multiple instances of that class, which can play different roles. For instance, have a look
at the implementation an Ape-X DQN agent :doc:`here </examples/atari/apex_dqn>`.


Object Reference
----------------

.. autoclass:: coax.Worker
