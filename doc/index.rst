coax
====

*Plug-n-play Reinforcement Learning in Python with OpenAI Gym and JAX*


.. image:: /_static/img/cartpole.gif
    :alt: Cartpole-v0 Environment
    :align: center


Coax is a modular Reinforcement Learning (RL) python package for solving OpenAI Gym environments
with JAX-based function approximators.


Install
-------

Coax is built on top of JAX, but it doesn't have an explicit dependence on the ``jax``
python package. The reason is that your version of ``jaxlib`` will depend on your CUDA version.

To install ``coax`` and ``jax`` together, please select the configuration that applies to your
setup.

.. raw:: html
    :file: versions.html


Alternatively, you could build ``jaxlib`` from source by following
`this guide <https://jax.readthedocs.io/en/latest/developer.html#building-from-source>`_.


Introduction
------------

Here's a short video that explains some design choices for **coax**.


.. raw:: html

    <div style="
          position: relative;
          padding-bottom: 56.25%;
          overflow: hidden;
          max-width: 100%;
          height: auto;">

      <iframe
          src="https://www.youtube.com/embed/r-aIbcGIc0s?rel=0"
          frameborder="0"
          allowfullscreen
          style="
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;">
      </iframe>

    </div>

|



RL concepts, not agents
-----------------------

The primary thing that sets coax apart from other packages is that is designed to align with the
core RL concepts, not with the high-level concept of an agent. This makes coax more modular and
user-friendly for RL researchers and practitioners.


You're in control
-----------------

Other RL frameworks often hide structure that you (the RL practitioner) are interested in. Most
notably, the **neural network architecture** of the function approximators is often hidden from you.
In coax, the network architecture takes center stage. You are in charge of defining their own
forward-pass function.

Another bit of structure that other RL frameworks hide from you is the main **training loop**. This
makes it hard to take an algorithm from paper to code. The design of coax is agnostic of the
details of your training loop. You decide how and when you update your function approximators.

To illustrate these points, we include the full working example that trains a simple Q-learning
agent in coax below.


Example
-------

We'll implement a simple q-learning agent on the non-slippery variant of the *FrozenLake*
environment, in which the agent must learn to navigate from the start state **S** to the goal state
**G**, without hitting the holes **H**, see grid below.

+---+---+---+---+
| S | F | F | F |
+---+---+---+---+
| F | H | F | H |
+---+---+---+---+
| F | F | F | H |
+---+---+---+---+
| H | F | F | G |
+---+---+---+---+

We start by defining our q-function. In coax, this is done by specifying a forward-pass
function:

.. code:: python

    import gym
    import coax
    import haiku as hk

    env = gym.make('FrozenLakeNonSlippery-v0')
    env = coax.wrappers.TrainMonitor(env)

    def func(S, is_training):
        values = hk.Linear(env.action_space.n, w_init=jnp.zeros)
        return values(S)  # shape: (batch_size, num_actions)


Note that if the action space is discrete, there are generally two ways of modeling a q-function:

.. math::

    (s,a)   &\ \mapsto\ q(s,a)\in\mathbb{R}    &\qquad  &(\text{type 1}) \\
    s       &\ \mapsto\ q(s,.)\in\mathbb{R}^n  &\qquad  &(\text{type 2})

where :math:`n` is the number of discrete actions. Type-1 q-functions may be defined for any action
space, whereas type-2 q-functions are specific to discrete actions. Coax accommodates both types of
q-functions. In this example, we're using a type-2 q-function.

Now that we defined our forward-pass function, we can create a q-function:

.. code:: python

    q = coax.Q(func, env)

    # example input
    s = env.observation_space.sample()
    a = env.action_space.sample()

    # example usage
    q(s, a)  # 0.
    q(s)     # array([0., 0., 0., 0.])


A function approximator :math:`q_\theta(s,a)` holds a collection of model parameters (weights),
denoted :math:`\theta`. These parameters are included in the q-function instance as:

.. code:: python

    q.params
    # frozendict({
    #   'linear': frozendict({
    #      'w': DeviceArray(shape=(16, 4), dtype=float32),
    #      'b': DeviceArray(shape=(4,), dtype=float32),
    #    }),
    # })

These :code:`q.params` are used internally when we call the function, e.g. :code:`q(s,a)`. The next
step is to create a policy, i.e. a function that maps states to actions. We'll use a simple
value-based policy:

.. code:: python

    # derive policy from q-function
    pi = coax.EpsilonGreedy(q, epsilon=1.0)  # we'll scale down epsilon later

    # sample action
    a = pi(s)

The action :code:`a` is an integer :math:`a\in\{0,1,2,3\}`, representing a single action. Now that
we have our policy, we can start doing episode roll-outs:

.. code:: python

    s = env.reset()

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # this is where we should update our q-function
        ...

        if done:
            break

        s = s_next


Of course, we can't expect our policy to do very well, because it hasn't been able to learn anything
from the reward signal :code:`r`. To do that, we need to create two more objects: a  **tracer** and
an **updater**. A *tracer* takes raw transition data and turns it into transition data can be
readily used by the *updater* to update our function approximator. In the example below we see how
this works in practice.

.. code:: python

    from optax import adam

    # tracer and updater
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
    qlearning = coax.td_learning.QLearning(q, optimizer=adam(0.02))


    for ep in range(500):
        pi.epsilon *= 0.99  # reduce exploration over time
        s = env.reset()

        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, info = env.step(a)

            # trace and update
            tracer.add(s, a, r, done)
            while tracer:
                transition_batch = tracer.pop()
                qlearning.update(transition_batch)

            if done:
                break

            s = s_next


    # [TrainMonitor|INFO] ep: 1,   T: 21,  G: 0,   avg_G: 0,   t: 20,  dt: 33.436ms
    # [TrainMonitor|INFO] ep: 2,   T: 42,  G: 0,   avg_G: 0,   t: 20,  dt: 2.504ms
    # [TrainMonitor|INFO] ep: 3,   T: 58,  G: 0,   avg_G: 0,   t: 15,  dt: 2.654ms
    # [TrainMonitor|INFO] ep: 4,   T: 72,  G: 0,   avg_G: 0,   t: 13,  dt: 2.670ms
    # [TrainMonitor|INFO] ep: 5,   T: 83,  G: 0,   avg_G: 0,   t: 10,  dt: 2.565ms
    # ...
    # [TrainMonitor|INFO] ep: 105, T: 1,020,   G: 0,   avg_G: 0.0868,  t: 5,   dt: 3.088ms
    # [TrainMonitor|INFO] ep: 106, T: 1,023,   G: 0,   avg_G: 0.0781,  t: 2,   dt: 3.154ms
    # [TrainMonitor|INFO] ep: 107, T: 1,035,   G: 1,   avg_G: 0.17,    t: 11,  dt: 3.401ms
    # [TrainMonitor|INFO] ep: 108, T: 1,044,   G: 0,   avg_G: 0.153,   t: 8,   dt: 2.432ms
    # [TrainMonitor|INFO] ep: 109, T: 1,057,   G: 1,   avg_G: 0.238,   t: 12,  dt: 2.439ms
    # [TrainMonitor|INFO] ep: 110, T: 1,065,   G: 1,   avg_G: 0.314,   t: 7,   dt: 2.428ms
    # ...
    # [TrainMonitor|INFO] ep: 495, T: 4,096,   G: 1,   avg_G: 1,   t: 6,   dt: 2.572ms
    # [TrainMonitor|INFO] ep: 496, T: 4,103,   G: 1,   avg_G: 1,   t: 6,   dt: 2.611ms
    # [TrainMonitor|INFO] ep: 497, T: 4,110,   G: 1,   avg_G: 1,   t: 6,   dt: 2.601ms
    # [TrainMonitor|INFO] ep: 498, T: 4,117,   G: 1,   avg_G: 1,   t: 6,   dt: 2.571ms
    # [TrainMonitor|INFO] ep: 499, T: 4,124,   G: 1,   avg_G: 1,   t: 6,   dt: 2.611ms


Getting Started
---------------

Have a look at the :doc:`Getting Started <examples/getting_started/install>` page to train your
first RL agent.

If this ain't your first rodeo, head over the examples listed :doc:`here <examples/index>`.


.. hidden tocs .....................................................................................

.. toctree::
    :caption: Examples
    :maxdepth: 1
    :hidden:

    examples/getting_started/index
    examples/stubs/index
    examples/linear_regression/index
    examples/atari/index
    examples/cartpole/index
    examples/frozen_lake/index
    examples/pendulum/index


.. toctree::
    :caption: Function Approximators
    :maxdepth: 1
    :hidden:

    coax/value_functions
    coax/policies
    coax/dynamics_models
    coax/proba_dists

.. toctree::
    :caption: Update Strategies
    :maxdepth: 1
    :hidden:

    coax/td_learning
    coax/policy_objectives
    coax/model_updaters
    coax/reward_tracing
    coax/experience_replay
    coax/value_losses
    coax/value_transforms
    coax/regularizers

.. toctree::
    :caption: Utilities
    :maxdepth: 1
    :hidden:

    coax/wrappers
    coax/worker
    coax/utils
    coax/envs

.. toctree::
    :caption: Editorial
    :maxdepth: 1
    :hidden:

    genindex
    release_notes

