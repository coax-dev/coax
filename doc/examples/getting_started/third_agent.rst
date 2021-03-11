**Steps:** [ :doc:`install <install>` | :doc:`jax <prereq_jax>` | :doc:`haiku <prereq_haiku>` | :doc:`q-learning <first_agent>` | :doc:`dqn <second_agent>` | *ppo* | :doc:`next_steps <next_steps>` ]

PPO on Pong
===========


In this example we build an even more sophisticated agent known as :doc:`PPO </examples/stubs/ppo>`
(`paper <https://arxiv.org/abs/1707.06347>`_).


.. image:: /_static/img/pong.gif
    :alt: Beating Atari 2600 Pong after a few hundred episodes.
    :align: center


You'll solve the **Pong** environment, in which the agent learns to beat its opponent at the famous
Atari video game.

This is the first example in which we use policy-based method (as opposed to a value-based method).
It introduces the concept of a *policy optimizer* and a *policy regularizer*.

Just as before, you may either hit the Google Colab button or download and run the script on your
local machine.

----

:download:`ppo.py </examples/atari/ppo.py>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/atari/ppo.ipynb

.. literalinclude:: /examples/atari/ppo.py
