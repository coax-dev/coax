**Steps:** [ :doc:`install <install>` | :doc:`jax <prereq_jax>` | :doc:`haiku <prereq_haiku>` | *q-learning* | :doc:`dqn <second_agent>` | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

Q-Learning on FrozenLake
========================


In this first reinforcement learning example we'll solve a simple grid world environment.


.. image:: /_static/img/frozen_lake_q.gif
    :alt: Non-Slippery FrozenLake solved
    :width: 132px
    :align: center


Our agent starts at the top left cell, labeled **S**. The goal of our agent is to find its way to
the bottom right cell, labeled **G**. The cells labeled **H** are *holes*, which the agent must
learn to avoid.

In this example, we'll implement a simple value-based agent, which we update using the
:doc:`q-learning </examples/stubs/qlearning>` algorithm.


.. raw:: html

    <div style="
          position: relative;
          padding-bottom: 56.25%;
          overflow: hidden;
          max-width: 100%;
          height: auto;">

      <iframe
          src="https://www.youtube.com/embed/eJTI08dH1WI?rel=0"
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


To run this, either hit the Google Colab button or download and run the script on your local
machine.

----

:download:`qlearning.py </examples/frozen_lake/qlearning.py>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/frozen_lake/qlearning.ipynb

.. literalinclude:: /examples/frozen_lake/qlearning.py
