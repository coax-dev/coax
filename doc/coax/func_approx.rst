FuncApprox Class
================

The :class:`coax.FuncApprox` class is central to the **coax**. It provides an
interface between a gym-style environment and function approximators like
:doc:`value functions <value_functions>` or a parametrized :doc:`policies
<policies>`.

The main purpose of the FuncApprox class is to accommodate custom function
approximators, combined with easy weight sharing between different
function-approximator components. To create a custom function approximator you
typically only need to override the :attr:`body <coax.FuncApprox.body>` method,
which implements feature extraction from the input state observations
:math:`s`. There are situations, however, in which you need more flexibility.
Therefore, any of the following methods may be overridden:

.. hlist::
    :columns: 2

    * :attr:`optimizer <coax.FuncApprox.optimizer>`
    * :attr:`body <coax.FuncApprox.body>`
    * :attr:`state_action_combiner <coax.FuncApprox.state_action_combiner>`
    * :attr:`action_preprocessor <coax.FuncApprox.action_preprocessor>`
    * :attr:`head_v <coax.FuncApprox.head_v>`
    * :attr:`head_pi <coax.FuncApprox.head_pi>`
    * :attr:`head_q1 <coax.FuncApprox.head_q1>`
    * :attr:`head_q2 <coax.FuncApprox.head_q2>`

All but the first (:attr:`optimizer <coax.FuncApprox.optimizer>`) are
function-approximator components, which are put together according to this flow
chart:

.. image:: /_static/img/func_approx_structure.svg
    :alt: Structure of FuncApprox components
    :width: 100%

Each component is defined using Haiku, see :doc:`Haiku documentation
<haiku:index>` for more details.

Finally, the :attr:`optimizer <coax.FuncApprox.optimizer>` method specifies the
:mod:`optix <jax.experimental.optix>` style optimizer used for updating the
model parameters (weights).

**Tip**: If you want to override FuncApprox methods, it's easiest to copy the
code from the default implementation and adjust it to your needs. In the
:ref:`Object Reference <object_reference>` below you'll find **[source]** links
that allow you to peek at the source code directly from here.


Example
-------

Here is an example of how one would override a functional-approximator
block (just the :attr:`body <coax.FuncApprox.body>` in this case):

.. code:: python

    import gym
    import jax
    import coax
    import haiku as hk

    class MLP(coax.FuncApprox):
        """ simple multi-layer perceptron """
        def body(self, S):
            seq = hk.Sequential([
                hk.Linear(32), jax.nn.relu,
                hk.Linear(16), jax.nn.relu,
            ])
            return seq(S)

    # instantiate function approximator
    env = gym.make(...)
    mlp = MLP(env)

    # define q-function
    q = coax.Q(mlp)
    pi = coax.EpsilonGreedy(q, epsilon=0.1)

    # or alternatively, a parametrized policy
    pi = coax.Policy(mlp)

    # run an episode
    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # do something
        ...

        if done:
            break

        s = s_next


.. _object_reference:

Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.FuncApprox

.. autoclass:: coax.FuncApprox
