FrozenLake
==========

In these notebooks we solve a *non-slippery* version of the
`FrozenLake <https://gymnasium.farama.org/environments/toy_text/frozen_lake/>`_ environment.

This is a very simple task, which is primarily used as a unit test for implementating new components
to the **coax** package.


.. toctree::
    :caption: Value-Based Agents

    SARSA <sarsa>
    Expected SARSA <expected_sarsa>
    Q-Learning <qlearning>
    Double Q-Learning <double_qlearning>

.. toctree::
    :caption: Policy-Based Agents

    REINFORCE <reinforce>
    A2C <a2c>
    PPO <ppo>
    DDPG <ddpg>
    TD3 <td3>

.. toctree::
    :caption: Distributional RL

    Stochastic SARSA <stochastic_sarsa>
    Stochastic Expected-SARSA <stochastic_expected_sarsa>
    Stochastic Q-Learning <stochastic_qlearning>
    Stochastic Double Q-Learning <stochastic_double_qlearning>
