**Steps:** [ :doc:`install <install>` | *jax* | :doc:`haiku <prereq_haiku>` | :doc:`q-learning <first_agent>` | :doc:`dqn <second_agent>` | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

Prerequisite: JAX
=================


The **coax** RL package is build on top of JAX, which is an alternative implementation of numpy that
includes auto-differentiation and hardware-specific acceleration. Think of it as Tensorflow or
Pytorch, but without the complications of a computation graph or eager execution.

In this example we briefly introduce the basics of JAX by implementing a linear regression model
from scratch.

.. raw:: html

    <div style="
          position: relative;
          padding-bottom: 56.25%;
          overflow: hidden;
          max-width: 100%;
          height: auto;">

      <iframe
          src="https://www.youtube.com/embed/aOsZdf9tiNQ?rel=0"
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


----

:download:`jax.py </examples/linear_regression/jax.py>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/main/doc/_notebooks/linear_regression/jax.ipynb

.. literalinclude:: /examples/linear_regression/jax.py
