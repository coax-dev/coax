Policies
========

There are generally two distinct ways of constructing a policy
:math:`\pi(a|s)`. One method uses a function approximator to parametrize a
state-action value function :math:`q_\theta(s,a)` and then derives a policy
from this q-function. The other method uses a function approximator to
parametrize the policy directly, i.e. :math:`\pi(a|s)=\pi_\theta(a|s)`. The
methods are called *value-based* methods and *policy gradient* methods,
respectively.


Parametrized policies
---------------------

Let's start with the policy-gradient style function approximator
:math:`\pi_\theta(a|s)`. This is implemented by :class:`coax.Policy`, which
uses a :class:`coax.FuncApprox` object, from which it uses the :attr:`body
<coax.FuncApprox.body>` and :attr:`head_pi <coax.FuncApprox.head_pi>` methods
for its forward-pass. The parametrization of the function approximator is
specified through :class:`coax.FuncApprox`, e.g.

.. code:: python

    import gym
    import jax
    import haiku as hk

    class MyFunc(coax.FuncApprox):
        """ simple MLP with 2 hidden layers """
        def body(self, S):
            seq = hk.Sequential([
                hk.Linear(24), jax.nn.relu,
                hk.Linear(16), jax.nn.relu,
            ])
            return seq(S)

    env = gym.make(...)
    func = MyFunc(env)
    pi = coax.Policy(func)

    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # update policy
        ...

        if done:
            break

        s = s_nest


Value-based policies
--------------------

Value-based policies are defined indirectly, via a q-function. Examples of
value-based policies are :class:`coax.EpsilonGreedy` (see example below) and
:class:`coax.BoltzmannPolicy`.


.. code:: python

    import gym
    import jax
    import haiku as hk

    class MyFunc(coax.FuncApprox):
        """ simple MLP with 2 hidden layers """
        def body(self, S):
            seq = hk.Sequential([
                hk.Linear(24), jax.nn.relu,
                hk.Linear(16), jax.nn.relu,
            ])
            return seq(S)

    env = gym.make(...)
    func = MyFunc(env)
    q = coax.Q(func)
    pi = coax.EpsilonGreedy(func, epsilon=0.1)

    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # update q-function
        ...

        if done:
            break

        s = s_nest


Random policy
-------------

The :class:`coax.RandomPolicy` doesn't depend on any function approximator. It
merely calls ``.sample()`` on the action space of the underlying gym
enviroment. This policy is particularly useful if you want to have a quick look
at an environment.


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.Policy
    coax.EpsilonGreedy
    coax.BoltzmannPolicy
    coax.RandomPolicy

.. autoclass:: coax.Policy
.. autoclass:: coax.EpsilonGreedy
.. autoclass:: coax.BoltzmannPolicy
.. autoclass:: coax.RandomPolicy
