==========================
coax: A Modular RL Package
==========================

*This is a draft of a blog post to be published on the MSR blog*


In this post we present **coax**, a new modular approach to reinforcement learning (RL). We motivate
our design choices. Before we do, we quickly introduce some key concepts in RL.


Agents
------

An *agent* is key concept at the heart of RL. We can think of an agent as a *program* that learns to
maximize some reward signal from experience. Many of the packages available today come with
predefined agents, which often have cryptic names such as DQN, DDPG, A2C, PPO, to name a few.

Although convenient for high-level reasoning, we argue that the concept of an agent is
*inconvenient* for organizing an RL framework. The reason is that it's too high-level.


The Status Quo
--------------

The current stage of the software ecosystem for RL may be characterized as adolescence. This is
particularly clear when compared to the software landscape for ordinary supervised learning.

**Two approaches.** Let's suppose that you want to build your own RL agent. Broadly speaking, there
are two options available to you. The first is to copy some existing code and then tweak it to fit
the new environment. The second approach, which seems to be most popular, is to write the entire
agent from scratch. Let's call these approaches *fork and hack* and *build from scratch*.

There are pros and cons for both approaches, but a downside that they share is that they both
involve code that isn't easily reused. The reason is that when you build an RL agent, you're
constantly making decisions on how you put different components together. These choices are then
hard-coded in your agent.

**Another approach: Frameworks and agents.** The two approaches we just mentioned aren't the full
story. There are packages out there that aim to provide reusability via a high-level API. The
problem with these packages is that they often use the concept of an *agent* to organize the API.

We mentioned earlier that an agent is essentially a type of program. Although it's possible to
represent programs as single classes, it's not the most convenient representation. The reason is
that there are too many choices to be made under the hood. Therefore, if we organize our API around
the concept of an agent, we end up with class definitions with a large number of configuration
settings. For instance, here's the function signature for a simple DQN [#dqn_paper]_ agent from
**tf-agents**:

.. code:: python

    tf_agents.agents.DqnAgent(
        time_step_spec, action_spec, q_network, optimizer,
        observation_and_action_constraint_splitter=None, epsilon_greedy=0.1,
        n_step_update=1, boltzmann_temperature=None, emit_log_probability=False,
        target_q_network=None, target_update_tau=1.0, target_update_period=1,
        td_errors_loss_fn=None, gamma=1.0, reward_scale_factor=1.0,
        gradient_clipping=None, debug_summaries=False, summarize_grads_and_vars=False,
        train_step_counter=None, name=None
    )

Another example might be the DQN implementation in **RLlib**, which has even more configuration
settings [#dqn_rllib]_.


A Fresh Start
-------------

With **coax** we aim to make a fresh start, with a simple and intuitive design. The key to this new
approach is *modularity*. If you build your own agent, you want to have control over the internal
components of the agent, see figure below.


.. figure:: https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg
    :alt: A Taxonomy of RL Algorithms
    :height: 400px
    :width: 600px
    :scale: 100%
    :align: center

    A Taxonomy of RL Algorithms, credits: https://spinningup.openai.com


For this reason, **coax** doesn't come with high-level agents. Instead, we offer so-called
:doc:`agent stubs </examples/stubs/index>` that show you how to put together your own agent.

**Under the hood.** Coax is based on the **JAX** [#jax_docs]_ project, which offers an alternative
implementation of numpy combined with automatic differentiation and hardware acceleration. It's
awesome and it's fast. As our neural network library we use is **Haiku** [#haiku_docs]_, which is an
intuitive library based on Sonnet (a lightweight alternative to keras).


Show me the code!
-----------------

Let's look at a specific example so that we get a feel for how the **coax** API works. We'll
implement a simple q-learning agent.

We start by defining our q-function. In **coax**, this is done by specifying a forward-pass
function:

.. code:: python

    import gym
    import coax
    import haiku as hk

    env = gym.make('FrozenLakeNonSlippery-v0')
    env = coax.wrappers.TrainMonitor(env)

    def func(S, is_training):
        values = hk.Sequential((
            lambda x: hk.one_hot(x, env.observation_space.n),
            hk.Linear(env.action_space.n, w_init=jnp.zeros)
        ))
        return values(S)  # shape: (batch_size, num_actions)


Note that if the action space is discrete, there are generally two ways of modeling a q-function:

.. math::

    (s,a)   &\ \mapsto\ q(s,a)\in\mathbb{R}    &\qquad  &(\text{type 1}) \\
    s       &\ \mapsto\ q(s,.)\in\mathbb{R}^n  &\qquad  &(\text{type 2})

where :math:`n` is the number of discrete actions. Type-1 q-functions may be defined for any action
space, whereas type-2 q-functions are specific to discrete actions. **Coax** accommodates both types
of q-functions. In this example, we're using a type-2 q-function.

Now that we defined our forward-pass function, we can create a q-function:

.. code:: python

    q = coax.Q(func, env.observation_space, env.action_space)

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
    pi = coax.EpsilonGreedy(q, epsilon=1.0)  # we'll scale this down later

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

To see more examples, head over to the documentation pages for a gentle introduction to the **coax**
RL package:

- https://coax.readthedocs.io


References
----------

.. [#dqn_paper]

    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

.. [#dqn_rllib]

    The DQN config in RLlib is too large to display here. See
    `DQN config <https://docs.ray.io/en/master/rllib-algorithms.html#dqn>`_ in the RLlib docs.

.. [#jax_docs]

    https://jax.readthedocs.io


.. [#haiku_docs]

    https://dm-haiku.readthedocs.io


.. references ......................................................................................


.. hrefs ...........................................................................................


