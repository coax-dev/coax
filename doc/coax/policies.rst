Policies
========

.. autosummary::
    :nosignatures:

    coax.Policy
    coax.EpsilonGreedy
    coax.BoltzmannPolicy
    coax.RandomPolicy

----

There are generally two distinct ways of constructing a policy :math:`\pi(a|s)`. One method uses a
function approximator to parametrize a state-action value function :math:`q_\theta(s,a)` and then
derives a policy from this q-function. The other method uses a function approximator to parametrize
the policy directly, i.e. :math:`\pi(a|s)=\pi_\theta(a|s)`. The methods are called *value-based*
methods and *policy gradient* methods, respectively.


A policy in **coax** is a function that maps state observations to actions. The example below shows
how to use a policy in a simple episode roll-out.

.. code:: python

    env = gym.make(...)

    s = env.reset()
    for t in range(max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        if done:
            break

        s = s_next

Some algorithms require us to collect the log-propensities along with the sampled actions. For this
reason, policies have the optional :code:`return_logp` flag:

.. code:: python

    a, logp = pi(s, return_logp=True)

The log-propensity represents :math:`\log\pi(a|s)`, which is a non-positive real-valued number. A
stochastic policy returns :code:`logp<0`, whereas a deterministic policy returns :code:`logp=0`.

As an aside, we note that **coax** policies have two more methods:

.. code:: python

    a = pi.mode(s)                   # same as pi(s), except 'sampling' greedily
    dist_params = pi.dist_params(s)  # distribution parameters, conditioned on s
    print(dist_params)               # in this example: categorical dist with n=3
    # {'logits': array([-0.5711, 1.0513 , 0.0012])}


Random policy
-------------

Before we discuss value-based policies and parametrized policies, let's discuss the simplest
possible policy first, namely :class:`coax.RandomPolicy`. This policy doesn't require any function
approximator. It simply calls :code:`env.action_space.sample()`. This policy may be useful for
creating simple benchmarks.

.. code:: python

    pi = coax.RandomPolicy(env)


Value-based policies
--------------------

Value-based policies are defined indirectly, via a :doc:`q-function <value_functions>`. Examples of
value-based policies are :class:`coax.EpsilonGreedy` (see example below) and
:class:`coax.BoltzmannPolicy`.

.. code:: python

    pi = coax.EpsilonGreedy(q, epsilon=0.1)
    pi = coax.BoltzmannPolicy(q, temperature=0.02)


Note that the hyperparameters :code:`epsilon` and :code:`temperature` may be updated at any time,
e.g.

.. code:: python

    pi.epsilon *= 0.99  # at the start of each epsiode


Parametrized policies
---------------------

Now that we've discussed value-based policies, let's start our discussion of parametrized
(learnable) policies. We provide three examples:

1. :ref:`Discrete actions <discrete>` (categorical dist)
2. :ref:`Continuous actions <continuous>` (normal dist)
3. :ref:`Composite actions <composite>`



.. _discrete:

**Discrete actions**

A common action space is :class:`Discrete <gym.spaces.Discrete>`. As an example, we'll take the
**CartPole** environment. To get started, let's generate some example data so that we know the
correct input/output format for our forward-pass function.

.. code:: python

    env = gym.make('CartPole-v0')

    data = coax.Policy.example_data(env)

    print(data)
    # ExampleData(
    #   inputs=Inputs(
    #     args=ArgsType2(
    #       S=array(shape=(1, 4), dtype=float32)
    #       is_training=True)
    #     static_argnums=(1,))
    #   output={
    #     'logits': array(shape=(1, 2), dtype=float32)})


Now, our task is to write a Haiku-style forward-pass function that generates this output given the
input. *To be clear, our task is not to recreate the exact values; the example data is only there to
give us an idea of the structure (shapes, dtypes, etc.).*

.. code:: python

    def func(S, is_training):
        logits = hk.Sequential((
            hk.Flatten(),
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(env.action_space.n, w_init=jnp.zeros)
        ))
        return {'logits': logits(S)}


    pi = coax.Policy(func, env)

    # example usage
    s = env.observation_space.sample()
    a = pi(s)
    print(a)  # 0 or 1

If something goes wrong and you'd like to debug the forward-pass function, here's an example of what
:attr:`coax.Policy.__init__` runs under the hood:

.. code:: python

    rngs = hk.PRNGSequence(42)
    transformed = hk.transform_with_state(func)
    params, function_state = transformed.init(next(rngs), *data.inputs.args)
    output, function_state = transformed.apply(params, function_state, next(rngs), *data.inputs.args)


.. _continuous:

**Continuous actions**

Besides discrete actions, we might wish to build an agent compatible with continuous actions. Here's
an example of how to create a valid policy function approximator for the **Pendulum** environment:

.. code:: python

    import coax
    import jax
    import haiku as hk
    from math import prod

    def func(S, is_training):
        shared = hk.Sequential((
            hk.Flatten(),
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
        ))
        mu = hk.Sequential((
            shared,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
            hk.Reshape(env.action_space.shape),
        ))
        logvar = hk.Sequential((
            shared,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
            hk.Reshape(env.action_space.shape),
        ))
        return {'mu': mu(S), 'logvar': logvar(S)}


    pi = coax.Policy(func, env)

    # example usage
    s = env.observation_space.sample()
    a = pi(s)

    print(a)
    # array([0.39267802], dtype=float32)


Note that if you're ever unsure what the correct input / output format is, you can always generate
some example data using the :func:`coax.Policy.example_data` helper (see example above).


.. _composite:

**Composite actions**

The **coax** package supports all action spaces that are supported by the `gym.spaces
<https://gym.openai.com/docs/#spaces>`_ API.

To illustrate the flexibility of the **coax** framework, here's an example of a composite action
space:

.. code:: python

    from collections import namedtuple
    from gym.spaces import Dict, Tuple, Box, Discrete, MultiDiscrete

    DummyEnv = namedtuple('DummyEnv', ('observation_space', 'action_space'))
    env = DummyEnv(
        Box(low=0, high=1, shape=(7,)),
        Dict({
            'foo': MultiDiscrete([4, 5]),
            'bar': Tuple((Box(low=0, high=1, shape=(2, 3)),)),
        }))

    data = coax.Policy.example_data(observation_space, action_space)
    print(data.output)
    # {'foo': ({'logits': DeviceArray([[-1.29,  0.34,  1.57,  1.88]], dtype=float32)},
    #          {'logits': DeviceArray([[-0.11, -0.35, -0.57,  2.51, 1.78]], dtype=float32)}),
    #  'bar': ({'logvar': DeviceArray([[[-0.11,  1.23,  0.12],
    #                                   [-0.35,  0.46,  0.73]]], dtype=float32),
    #           'mu': DeviceArray([[[-0.35, -0.37, -0.67],
    #                               [-0.44, -0.71,  0.45]]], dtype=float32)},)}

Thus, if we ensure that our forward-pass function outputs this format, we can sample actions in
precisely the same way as we've done before. For example, here's a compatible forward-pass function:

.. code:: python

    def func(S, is_training):
        return {
            'foo': ({'logits': hk.Linear(4)(S)},
                    {'logits': hk.Linear(5)(S)}),
            'bar': ({'mu': hk.Linear(6)(S).reshape(-1, 2, 3),
                     'logvar': hk.Linear(6)(S).reshape(-1, 2, 3)},),
        }

    pi = coax.Policy(func, env)

    # example usage:
    s = observation_space.sample()
    a, logp = pi(s, return_logp=True)
    assert a in action_space

    print(logp)  # -8.647176
    print(a)
    # {'foo': array([2, 4]),
    #  'bar': (array([[0.18, 0.57, 0.38],
    #                 [0.81, 0.21, 0.67]], dtype=float32),)}



Object Reference
----------------

.. autoclass:: coax.Policy
.. autoclass:: coax.EpsilonGreedy
.. autoclass:: coax.BoltzmannPolicy
.. autoclass:: coax.RandomPolicy
