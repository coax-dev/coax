Agent Stubs
===========

The **coax** package doesn't include full-fledged agents. The reason is that it
is designed for RL practitioners who want to create their own agents using a
modular API.

For those who would like to have a starting point for creating pre-existing
agents such as DQN or PPO we have collected simple **Agent Stubs** here. These
stubs are partially complete python scripts that show the general structure of
an agent.


.. toctree::
    :maxdepth: 1
    :caption: Value-Based Agents

    sarsa
    qlearning
    dqn


.. toctree::
    :maxdepth: 1
    :caption: Policy-Based Agents

    reinforce
    a2c
    ppo
    ddpg

