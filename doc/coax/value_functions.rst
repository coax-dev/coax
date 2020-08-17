Value Functions
===============

The are two kinds of value functions, state value functions :math:`v(s)` and state-action value
functions (or q-functions) :math:`q(s,a)`. The state value function evaluates the expected
(discounted) return, defined as:

.. math::

    v(s)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots \,\Big|\, S_t=s
    \right\}

The operator :math:`\mathbb{E}_t` takes the expectation value over all transitions (indexed by
:math:`t`). The :math:`v(s)` function is implemented by the :class:`coax.V` class. The state-action
value is defined in a similar way:

.. math::

    q(s,a)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots \,\Big|\, S_t=s, A_t=a
    \right\}

This is implemented by the :class:`coax.Q` class.


v(s)
----

In this example we see how to construct a valid state value function :math:`v(s)`. We'll start by
creating some example data, which allows us inspect the correct input/output format.

.. code:: python

    import coax
    import gym

    env = gym.make('CartPole-v0')
    data = coax.V.example_data(env.observation_space)

    print(data)
    # ExampleData(
    #   inputs=Inputs(
    #     args=Args(
    #       S=DeviceArray([[0.7222302 , 0.6115424 , 0.18975885, 0.7168227 ]], dtype=float32),
    #       is_training=True),
    #     static_argnums=(1,)),
    #   output=DeviceArray([1.60552], dtype=float32))

From this we may define our Haiku-style forward-pass function:

.. code:: python

    import jax
    import jax.numpy as jnp
    import haiku as hk

    def func(S, is_training):
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros), jnp.ravel
        ))
        return seq(S)


    v = coax.V(func, env.observation_space)

    # example usage
    s = env.observation_space.sample()
    print(v(s))  # 0.0


q(s, a)
-------

In this example we see how to construct a valid state-action value function :math:`q(s,a)`. let's
create some example data again.

.. code:: python

    import coax
    import gym

    env = gym.make('CartPole-v0')
    data = coax.Q.example_data(env.observation_space)

    print(data.type1)
    # ExampleData(
    #   inputs=Inputs(
    #     args=Args(
    #       S=DeviceArray([[0.7476373 , 0.12747586, 0.41887903, 0.0605397 ]], dtype=float32),
    #       A=DeviceArray([[0., 1.]], dtype=float32),
    #       is_training=True),
    #     static_argnums=(2,)),
    #   output=DeviceArray([0.74104834], dtype=float32))

    print(data.type2)
    # ExampleData(
    #   inputs=Inputs(
    #     args=Args(
    #       S=DeviceArray([[0.84571934, 0.38024798, 0.27534822, 0.7612812 ]], dtype=float32),
    #       is_training=True),
    #     static_argnums=(1,)),
    #   output=DeviceArray([[ 1.2767161 , -0.05539829]], dtype=float32))

Note that there are **two types** of modeling a q-function:

.. math::

    (s,a)   &\ \mapsto\ q(s,a)\in\mathbb{R}    &\qquad  &(\text{type 1}) \\
    s       &\ \mapsto\ q(s,.)\in\mathbb{R}^n  &\qquad  &(\text{type 2})

where :math:`n` is the number of discrete actions. Note that type-2 q-functions are only
well-defined for discrete action spaces, whereas type-1 q-functions may be defined for any action
space.

Let's first define our **type-1** forward-pass function:

.. code:: python

    import jax
    import jax.numpy as jnp
    import haiku as hk

    def func_type1(S, A, is_training):
        """ (s,a) -> q(s,a) """
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros), jnp.ravel
        ))
        X = jnp.concatenate((S, A), axis=-1)
        return seq(X)


    q = coax.Q(func_type1, env.observation_space, env.action_space)

    # example usage
    s = env.observation_space.sample()
    a = env.action_space.sample()
    print(q(s, a))  # 0.0
    print(q(s))     # array([0., 0.])


Alternatively, a **type-2** forward-pass function might be:

.. code:: python

    def func_type2(S, is_training):
        """ s -> q(s,.) """
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(env.action_space.n, w_init=jnp.zeros)
        ))
        return seq(S)


    q = coax.Q(func_type2, env.observation_space, env.action_space)

    # example usage
    s = env.observation_space.sample()
    a = env.action_space.sample()
    print(q(s, a))  # 0.0
    print(q(s))     # array([0., 0.])


If something goes wrong and you'd like to debug the forward-pass function, here's an example of what
:attr:`coax.Q.__init__` runs under the hood:

.. code:: python

    rngs = hk.PRNGSequence(42)
    func = hk.transform_with_state(func_type2)
    params, function_state = func.init(next(rngs), *data.type2.inputs)
    output, function_state = func.apply(params, function_state, next(rngs), *data.type2.inputs)


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.V
    coax.Q

.. autoclass:: coax.V
.. autoclass:: coax.Q
