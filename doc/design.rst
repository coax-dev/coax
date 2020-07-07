======
Design
======

In this document we lay out the design decisions for the **coax** package.

One of the main ways that JAX differs from keras and pytorch is that it uses a more functional-style
design pattern, as opposed to the heavily object-oriented style design.

The nice thing about using an object-oriented design pattern is that you can stuff some state in any
object that you define. In contrast, JAX encourages a style in which you to pass on most of the
state from function to function. This may feel clunky, but it does allow for efficient
JIT-compilable chains of operations.

Although it's nice for the user to have full control over the state of the model weights at every
moment, it does make the process somewhat cumbersome and less user-friendly.

The **coax** package slightly breaks JAX's functional design pattern. This is done deliberately and
carefully. The way **coax** deviates from the functional design is by storing state in instances of
the :class:`coax.FuncApprox` class.


.. contents::
    :local:


The API
=======

The main goal of the **coax** package is to align the code with the underlying theoretical concepts.
The way the package is designed should follow the thought pattern of someone wishing to apply
reinforcement learning. Below we'll show an example of such a process (thoughts are printed in
*italic*).

*Let's start with an MDP.*

.. code:: python

    >>> env = gym.make('CartPole-v0')


*What does the state observation look like?*

.. code:: python

    >>> s = env.reset()
    >>> s
    array([ 0.0030509 ,  0.04395586, -0.04229512,  0.0231472 ])


*In fact, what does a transition look like?*

.. code:: python

    >>> tr = coax.get_transition(env)
    >>> tr
    +----------------------------------------------------------------------+
    | TransitionSingle                                                     |
    +--------+-------------------------------------------------------------+
    |    s   | array([ 0.01502608,  0.04071553,  0.01393078, -0.02459852]) |
    +--------+-------------------------------------------------------------+
    |    a   | 1                                                           |
    +--------+-------------------------------------------------------------+
    |    r   | 1.0                                                         |
    +--------+-------------------------------------------------------------+
    |  done  | False                                                       |
    +--------+-------------------------------------------------------------+
    |  info  | {}                                                          |
    +--------+-------------------------------------------------------------+
    | s_next | array([ 0.0158404 ,  0.23563496,  0.0134388 , -0.31285378]) |
    +--------+-------------------------------------------------------------+


*Let's see how well a random policy does here.*

.. code:: python

    >>> pi = coax.RandomPolicy(env)
    >>> coax.render_episode(env, pi)


.. image:: /_static/img/cartpole_random.gif
    :alt: Cartpole with random policy
    :width: 400px


*Now let's train an agent. We'll start with a simple function approximator.*

.. code:: python

    class MLP(coax.FuncApprox):
        def body(self):
            return stax.serial(Dense(4), Relu, Dense(4))

    # a simple multi-layer perceptron
    mlp = MLP(env, lr=0.01)

    # which we use as our Q-function
    q = coax.Q(mlp, gamma=0.9)

    # and we derive our policy from our Q-function
    pi = coax.EpsilonGreedy(q, epsilon=0.1)


*Now let's see if this agent can learn. Let's use Q-learning updates.*

.. code:: python

    # this is how we turn the rewards into a learning signal
    cache = caox.NStepCache(env, n=1, gamma=0.9)

    # and this is how specify how to update our q-function
    qlearning = coax.td_learning.QLearning(q)

    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # this is where the agent learns
        while cache:
            transition_batch = cache.pop()
            qlearning.update(transition_batch)

        if done:
            break

        s = s_next


*Actually, let's run multiple episodes.*

.. code:: python

    k = 0  # number of consecutive successes

    for ep in range(100):
        pi.epsilon = 0.1 if ep < 10 else 0.01  # exploration schedule

        s = env.reset()

        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, info = env.step(a)

            # this is where the agent learns
            while cache:
                transition_batch = cache.pop()
                qlearning.update(transition_batch)

            if done:
                if t == env.spec.max_episode_steps - 1:
                    k += 1
                    print(f"{k} successes")
                else:
                    k = 0
                    print(f"failed after {t} steps")
                break

            s = s_next

        # early-stopping
        if k == 10:
            break


*Let's look at how well this agent does now.*

.. code:: python

    pi.epsilon = 0
    coax.render_episode(env, pi)


.. image:: /_static/img/cartpole.gif
    :alt: Cartpole with learned policy
    :width: 400px


The FuncApprox class
====================

This is the object that carries all state, which includes model weights and optimizer state as well
as functions that prescribe operations specific to the environment and/or function-approximator
settings.

The stateful attributes are:

- :attr:`state <coax.FuncApprox.state>`
- :attr:`apply_funcs <coax.FuncApprox.apply_funcs>`

Below we explain what each of these attributes are. We will reference an object called ``func``,
which is instantiated as:

.. code:: python

    import coax
    func = coax.FuncApprox(env)


Components
----------
A function approximator consists of a collection of components, which are linked together according
to the flow chart below.


.. image:: /_static/img/func_approx_structure.svg
    :alt: Structure of FuncApprox components
    :width: 100%


FuncApprox.apply_funcs
----------------------

This attribute holds the collection of forward-pass functions, one for each component, i.e.

.. code:: python

    func.apply_funcs = {
        'body':    <apply_func>,                # state observation preprocessor
        'head_pi': <apply_func>,                # policy head
        'head_v':  <apply_func>,                # state value head
        'head_q1': <apply_func>,                # type-1 state-action value head
        'head_q2': <apply_func>,                # type-2 state-action value head
        'action_preprocessor':   <apply_func>,  # action preprocessor (stateless)
        'action_postprocessor':  <apply_func>,  # action postprocessor (stateless)
        'state_action_combiner': <apply_func>,  # combiner for type-1 q-function
    }


Each forward-pass function has the signature:

.. code:: python

    apply_func: params, function_state, rng, *inputs -> output


where the output may be any :doc:`pytree <pytrees>`. Besides the actual function inputs ``*inputs``,
these apply-functions require some additional input:

.. code:: python

    params          # model parameters (weights)
    function_state  # internal state of the forward-pass function
    rng             # jax pseudo-random number generator key

The first two inputs are stored in the :attr:`func.state <coax.FuncApprox.state>` attribute. The
third input can be generated easily by accessing the :attr:`func.rng <coax.FuncApprox.rng>`
property.


FuncApprox.state
----------------

The state of the function approximator is stored in the :attr:`state <coax.FuncApprox.state>`
attribute:

.. code:: python

    func.state = {
        'body':                  {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'head_pi':               {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'head_v':                {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'head_q1':               {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'head_q2':               {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'action_preprocessor':   {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'action_postprocessor':  {'params': ..., 'function_state': ..., 'optimizer_state': ...},
        'state_action_combiner': {'params': ..., 'function_state': ..., 'optimizer_state': ...},
    }


Each key in the :attr:`state <coax.FuncApprox.state>` dict above corresponds to the method by the
same name. Each of these methods is a Haiku-style function. It is very easy to replace a component
by your own custom implementation. All you need to do is create a class that inherits from
:class:`coax.FuncApprox` and the overrides one or more methods.


Value Functions and Updateable Policies
=======================================

These objects are the ones that the end user will mostly interact with. They should be intuitive and
easy to use. In terms of implementation, they wrap a :class:`FuncApprox <coax.FuncApprox>` object in
away that puts the necessary components together.

For instance, a state value function is defined via:

.. code:: python

    func = coax.FuncApprox(env)
    v = coax.V(func)

This allows us to evaluate a state observation :math:`s` by calling ``v(s)``. Under the hood, ``v``
uses its own :attr:`v.apply_func <coax.V.apply_func>`, which is a JIT-compiled (pure) function that
ties together the different function-approximator components (in this case :attr:`body
<coax.FuncApprox.body>` and :attr:`head_v <coax.FuncApprox.head_v>`).


Weight Sharing
--------------

Weight sharing is trivally easy in **coax**. You just point your functions to the same underlying
function approximator:

.. code:: python

    # actor-critic with weight sharing
    func = coax.FuncApprox(env)
    pi = coax.Policy(func)
    v = coax.V(func)

    # actor-critic without weight sharing
    func_pi = coax.FuncApprox(env)
    func_v = func_pi.copy()  # creates a deepcopy
    pi = coax.Policy(func_pi)
    v = coax.V(func_v)


Note that weight sharing typically does require that you add more structure to the different heads
(:attr:`head_pi <coax.FuncApprox.head_pi>` and :attr:`head_v <coax.FuncApprox.head_v>`). The reason
is that the default heads typically consist of a single :class:`hk.Linear <haiku.Linear>` layer,
relying on :attr:`body <coax.FuncApprox.body>` to do the heavy lifting.
