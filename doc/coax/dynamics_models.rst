Dynamics Models
===============

Model-based methods make use of a dynamics model :math:`p(s'|s,a)`, which is a probability
distribution over successor states :math:`s'`, conditioned on taking action :math:`a` from state
:math:`s`.

Coax allows you to define your own dynamics model with a function approximator, similar to how we
define :doc:`value functions <value_functions>` and :doc:`policies <policies>`. In particular, a
dynamics model is represented by a *stochastic* function, just like parametrized policies
:math:`\pi_\theta(a|s)`. This means that the forward-pass function returns distribution parameters
:math:`\varphi` that depend on the input state-action pair, i.e. :math:`\varphi_\theta(s,a)`. The
most common case is where the observation space is a :class:`Box <gym.spaces.Box>`, which means that
the distribution parameters are the parameters of a Gaussian distribution,
:math:`\varphi_\theta(s,a)=(\mu_\theta(s,a), \Sigma_\theta(s,a))`.


Example
-------

In this example we see how to construct a valid dynamics model :math:`p(s'|s,a)`. Let's create some
example data.

.. code:: python

    import coax
    import gym

    env = gym.make('CartPole-v0')
    data = coax.DynamicsModel.example_data(env.observation_space, env.action_space)

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
        S = check_onehot(S)
        X = jax.vmap(jnp.kron)(S, A)
        mu = S + seq(X)
        return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}


    p = coax.DynamicsModel(func_type1, env.observation_space, env.action_space)

    # example usage
    s = env.reset()
    a = env.action_space.sample()

    print(s)        # [ 0.008, 0.021, -0.037, 0.032]
    print(p(s, a))  # [-0.015, 0.067, -0.035, 0.029]
    print(p(s))     # [[-0.012, 0.064, -0.039, 0.041], [ 0.022, 0.048, -0.039, 0.027]]


Alternatively, a **type-2** forward-pass function might be:

.. code:: python

    def func_type2(S, is_training):
        """ s -> q(s,.) """
        output_shape = (env.action_space.n, *env.observation_space.shape)
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(prod(output_shape), w_init=jnp.zeros),
            hk.Reshape(output_shape),
        ))
        S = check_onehot(S)
        mu = S + seq(S)
        return {'mu': mu, 'logvar': jnp.full_like(mu, -10)}


    p = coax.DynamicsModel(func_type2, env.observation_space, env.action_space)

    # example usage
    s = env.reset()
    a = env.action_space.sample()

    print(s)        # [ 0.004,  0.041,  0.043, -0.015]
    print(p(s, a))  # [-0.024,  0.067,  0.042,  0.011]
    print(p(s))     # [[-0.014, -0.102,  0.041, -0.052], [0.007, -0.065, 0.044, 0.102]]





If something goes wrong and you'd like to debug the forward-pass function, here's an example of what
:attr:`coax.Dynamics.__init__` runs under the hood:

.. code:: python

    rngs = hk.PRNGSequence(42)
    func = hk.transform_with_state(func_type2)
    params, function_state = func.init(next(rngs), *data.type2.inputs)
    output, function_state = func.apply(params, function_state, next(rngs), *data.type2.inputs)


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.DynamicsModel

.. autoclass:: coax.DynamicsModel
