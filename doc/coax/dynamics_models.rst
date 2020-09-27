Dynamics Models
===============

Model-based methods make use of models that estimate the dynamics of transitions in a Markov
decision process. In coax we offers two types of such models: a **dynamics model** :math:`p(s'|s,a)`
and a **reward model** :math:`p(r|s,a)`, where  :math:`s'` is a successor state and :math:`r` is an
immediate reward. Both distributions are conditioned on taking action :math:`a` from state
:math:`s`.

Coax allows you to define your own dynamics model with a function approximator, similar to how we
define :doc:`value functions <value_functions>` and :doc:`policies <policies>`. A dynamics model is
represented by a *stochastic* function, just like parametrized policies :math:`\pi_\theta(a|s)`.
This means that the forward-pass function returns distribution parameters :math:`\varphi` that
depend on the input state-action pair, i.e. :math:`\varphi_\theta(s,a)`. The most common case is
where the observation space is a :class:`Box <gym.spaces.Box>`, which means that the distribution
parameters are the parameters of a Gaussian distribution,
:math:`\varphi_\theta(s,a)=(\mu_\theta(s,a), \Sigma_\theta(s,a))`.


Dynamics model
--------------

In this example we see how to construct a valid dynamics model :math:`p(s'|s,a)`. Let's create some
example data.

.. code:: python

    import coax
    import gym

    env = gym.make('CartPole-v0')
    data = coax.StochasticTransitionModel.example_data(env.observation_space, env.action_space)

    print(data.type1)
    # ExampleData(
    #   inputs=Inputs(
    #     args=ArgsType1(
    #       S=array(shape=(1, 4), dtype=float32)
    #       A=array(shape=(1, 2), dtype=float32)
    #       is_training=True)
    #     static_argnums=(2,))
    #   output={
    #     'logvar': array(shape=(1, 4), dtype=float32)
    #     'mu': array(shape=(1, 4), dtype=float32)})

    print(data.type2)
    # ExampleData(
    #   inputs=Inputs(
    #     args=ArgsType2(
    #       S=array(shape=(1, 4), dtype=float32)
    #       is_training=True)
    #     static_argnums=(1,))
    #   output={
    #     'logvar': array(shape=(1, 2, 4), dtype=float32)
    #     'mu': array(shape=(1, 2, 4), dtype=float32)})


Note that, similar to q-functions, there are **two types** of handling a discrete action space:

.. math::

    (s,a)   &\ \mapsto\ p(s'|s,a)  &\qquad  &(\text{type 1}) \\
    s       &\ \mapsto\ p(s'|s,.)  &\qquad  &(\text{type 2})

A type-2 model essentially returns a vector of distributions of size :math:`n`, which is the number
of discrete actions. Note that type-2 models are only well-defined for discrete action spaces,
whereas type-1 models may be defined for any action space.

Let's first define our **type-1** forward-pass function:

.. code:: python

    import jax
    import jax.numpy as jnp
    import haiku as hk
    from numpy import prod

    def func_type1(S, A, is_training):
        """ (s,a) -> p(s'|s,a) """
        output_shape = (env.action_space.n, *env.observation_space.shape)
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(prod(output_shape), w_init=jnp.zeros),
            hk.Reshape(output_shape),
        ))
        X = jax.vmap(jnp.kron)(S, A)
        mu = S + seq(X)
        return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}


    p = coax.StochasticTransitionModel(func_type1, env)

    # example usage
    s = env.reset()
    a = env.action_space.sample()

    print(s)        # [ 0.008, 0.021, -0.037, 0.032]
    print(p(s, a))  # [-0.015, 0.067, -0.035, 0.029]
    print(p(s))     # [[-0.012, 0.064, -0.039, 0.041], [ 0.022, 0.048, -0.039, 0.027]]


Alternatively, a **type-2** forward-pass function might be:

.. code:: python

    def func_type2(S, is_training):
        """ s -> p(s'|s,.) """
        output_shape = (env.action_space.n, *env.observation_space.shape)
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(prod(output_shape), w_init=jnp.zeros),
            hk.Reshape(output_shape),
        ))
        mu = S + seq(S)
        return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}


    p = coax.StochasticTransitionModel(func_type2, env)

    # example usage
    s = env.reset()
    a = env.action_space.sample()

    print(s)        # [ 0.004,  0.041,  0.043, -0.015]
    print(p(s, a))  # [-0.024,  0.067,  0.042,  0.011]
    print(p(s))     # [[-0.014, -0.102,  0.041, -0.052], [0.007, -0.065, 0.044, 0.102]]


If something goes wrong and you'd like to debug the forward-pass function, here's an example of what
the constructor runs under the hood:

.. code:: python

    rngs = hk.PRNGSequence(42)
    func = hk.transform_with_state(func_type2)
    params, function_state = func.init(next(rngs), *data.type2.inputs)
    output, function_state = func.apply(params, function_state, next(rngs), *data.type2.inputs)



Reward model
------------

In this example we see how to construct a valid reward model :math:`p(r|s,a)`. Let's create some
example data.

.. code:: python

    import coax
    import gym

    env = gym.make('CartPole-v0')
    data = coax.RewardModel.example_data(env.observation_space, env.action_space, env.reward_range)

    print(data.type1)
    # ExampleData(
    #   inputs=Inputs(
    #     args=ArgsType1(
    #       S=array(shape=(1, 4), dtype=float32)
    #       A=array(shape=(1, 2), dtype=float32)
    #       is_training=True)
    #     static_argnums=(2,))
    #   output={
    #     'logvar': array(shape=(1,), dtype=float32)
    #     'mu': array(shape=(1,), dtype=float32)})

    print(data.type2)
    # ExampleData(
    #   inputs=Inputs(
    #     args=ArgsType2(
    #       S=array(shape=(1, 4), dtype=float32)
    #       is_training=True)
    #     static_argnums=(1,))
    #   output={
    #     'logvar': array(shape=(1, 2), dtype=float32)
    #     'mu': array(shape=(1, 2), dtype=float32)})


Again, there are **two types** of handling a discrete action space:

.. math::

    (s,a)   &\ \mapsto\ p(r|s,a)  &\qquad  &(\text{type 1}) \\
    s       &\ \mapsto\ p(r|s,.)  &\qquad  &(\text{type 2})

A type-2 model essentially returns a vector of distributions of size :math:`n`, which is the number
of discrete actions. Note that type-2 models are only well-defined for discrete action spaces,
whereas type-1 models may be defined for any action space.

Let's first define our **type-1** forward-pass function:

.. code:: python

    import jax
    import jax.numpy as jnp
    import haiku as hk
    from numpy import prod

    def func_type1(S, A, is_training):
        """ (s,a) -> p(r|s,a) """
        mu = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros), jnp.ravel,
        ))
        logvar = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros), jnp.ravel,
        ))
        X = jax.vmap(jnp.kron)(S, A)
        return {'mu': mu(X), 'logvar': logvar(X)}


    r = coax.RewardModel(func_type1, env)

    # example usage
    s = env.reset()
    a = env.action_space.sample()

    print(r(s, a))  # 4.650
    print(r(s))     # [0.865, -5.022]


Alternatively, a **type-2** forward-pass function might be:

.. code:: python

    def func_type2(S, is_training):
        """ s -> p(r|s,.) """
        mu = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(env.action_space.n, w_init=jnp.zeros),
        ))
        logvar = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(env.action_space.n, w_init=jnp.zeros),
        ))
        return {'mu': mu(S), 'logvar': logvar(S)}


    r = coax.RewardModel(func_type2, env)

    # example usage
    s = env.reset()
    a = env.action_space.sample()

    print(r(s, a))  # -3.593
    print(r(s))     # [9.575, 1.974]


If something goes wrong and you'd like to debug the forward-pass function, here's an example of what
the constructor runs under the hood:

.. code:: python

    rngs = hk.PRNGSequence(42)
    func = hk.transform_with_state(func_type2)
    params, function_state = func.init(next(rngs), *data.type2.inputs)
    output, function_state = func.apply(params, function_state, next(rngs), *data.type2.inputs)


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.StochasticTransitionModel
    coax.RewardModel

.. autoclass:: coax.StochasticTransitionModel
.. autoclass:: coax.RewardModel
