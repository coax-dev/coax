**Steps:** [ *install* | :doc:`jax <prereq_jax>` | :doc:`haiku <prereq_haiku>` | :doc:`q-learning <first_agent>` | :doc:`dqn <second_agent>` | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

Install
=======

Coax is built on top of JAX, but it doesn't have an explicit dependence on the ``jax``
python package. The reason is that your version of ``jaxlib`` will depend on your CUDA version.

To install ``jax``, please have a look at the instructions: https://github.com/google/jax#installation

Once ``jax`` and ``jaxlib`` are installed, you can install **coax** simple by running:

.. code::

    $ pip install coax

Or, alternatively, to install **coax** from the latest branch on github:

.. code::
    
    $ pip install git+https://github.com/coax-dev/coax.git@main
