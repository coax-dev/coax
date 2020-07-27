**Steps:** [ :doc:`install <install>` | :doc:`jax <prereq_jax>` | *haiku* | :doc:`q-learning <first_agent>` | :doc:`dqn <second_agent>` | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

Prerequisite: Haiku
===================


In this example we continue the :doc:`introduction of JAX <prereq_jax>` by implementing the same
linear regression model but this this time using Haiku instead of bare-metal JAX.

.. raw:: html

    <div style="
          position: relative;
          padding-bottom: 56.25%;
          overflow: hidden;
          max-width: 100%;
          height: auto;">

      <iframe
          src="https://www.youtube.com/embed/eyCuUPeALVg?rel=0"
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

:download:`haiku.py </examples/linear_regression/haiku.py>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/linear_regression/haiku.ipynb

.. literalinclude:: /examples/linear_regression/haiku.py
