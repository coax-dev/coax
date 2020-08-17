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

    env = gym.make('FrozenLakeNonSlippery-v0')
    env = coax.wrappers.TrainMonitor(env)



.. code:: python

    import gym
    import coax
    import jax.numpy as jnp
    import haiku as hk
    from jax.experimental.optix import adam


    # pick environment
    env = gym.make(...)


    def func(S, A, is_training):
        """ forward pass with 3 hidden layers """
        seq = hk.Sequential((
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros), jnp.ravel
        ))

        X = jnp.concatenate((S, A), axis=-1)
        return seq(X)


    # function approximator
    q = coax.Q(func, env.observation_space, env.action_space)
    pi = coax.EpsilonGreedy(q, epsilon=0.1)


    # specify how to update q-function
    qlearning = coax.td_learning.QLearning(q, optimizer=adam(0.02))


    # specify how to trace the transitions
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)


    for ep in range(100):
        pi.epsilon = ...  # exploration schedule
        s = env.reset()

        for t in range(env.spec.max_episode_steps):
            a = pi(s)
            s_next, r, done, info = env.step(a)

            # "trace" raw transition
            tracer.add(s, a, r, done)

            # update
            while tracer:
                transition_batch = tracer.pop()
                qlearning.update(transition_batch)

            if done:
                break

            s = s_next


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


