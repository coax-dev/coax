**Steps:** [ *install* | :doc:`jax <prereq_jax>` | :doc:`haiku <prereq_haiku>` | :doc:`q-learning <first_agent>` | :doc:`dqn <second_agent>` | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

Install
=======

Coax is built on top of JAX, but it doesn't have an explicit dependence on the ``jax``
python package. The reason is that your version of ``jaxlib`` will depend on your CUDA version.

To install ``coax`` and ``jax`` together, please select the configuration that applies to your
setup.

----

.. raw:: html
    :file: ../../versions.html


Alternatively, you may also choose build ``jaxlib`` from source by following
`this guide <https://jax.readthedocs.io/en/latest/developer.html#building-from-source>`_.
