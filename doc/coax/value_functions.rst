Value Functions
===============

The are two kinds of value functions, state value functions :math:`v(s)` and
state-action value functions (or q-functions) :math:`q(s,a)`. The state value
function evaluates the expected (discounted) return, defined as:

.. math::

    v(s)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots \,\Big|\, S_t=s
    \right\}

The operator :math:`\mathbb{E}_t` takes the expectation value over all
transitions (indexed by :math:`t`). The :math:`v(s)` function is implemented by
the :class:`V <coax.V>` class. The state-action value is defined in a similar
way:

.. math::

    q(s,a)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots \,\Big|\, S_t=s, A_t=a
    \right\}

This is defined :class:`Q <coax.Q>`.

One thing to note about the q-function is that there are two distinct ways to
represent it with a function approximator. In **coax** we call these *type-I*
and *type-II* q-functions, where a type-I q-function models the value function
as:

.. math::

    (s, a) \mapsto q(s,a)\ \in\ \mathbb{R}  \qquad (type-I)

whereas a type-II models the value function as:

.. math::

    s \mapsto q(s,.)\ \in\ \mathbb{R}^n  \qquad (type-II)

where :math:`n` is the number of actions. The type is specified by setting the
:class:`Q(func, qtype=1) <coax.Q>` or :class:`Q(func, qtype=2) <coax.Q>`.

Note that a type-II q-function is only well-defined for discrete actions
spaces.




Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.V
    coax.Q

.. autoclass:: coax.V
.. autoclass:: coax.Q
