coax
====

*Plug-n-play Reinforcement Learning in Python with OpenAI Gym and JAX*


.. image:: /_static/img/cartpole.gif
    :alt: Cartpole-v0 Environment
    :align: center


Create simple, reproducible RL solutions with OpenAI gym environments and JAX-based function
approximators.

Install
=======

**coax** is built on top of JAX, but it doesn't have an explicit dependence on the ``jax``
python package. The reason is that your version of ``jaxlib`` will depend on your CUDA version.

To install ``coax`` and ``jax`` together, please select the configuration that applies to your
setup.

.. raw:: html
    :file: versions.html


Alternatively, you may also choose build ``jaxlib`` from source by following
`this guide <https://jax.readthedocs.io/en/latest/developer.html#building-from-source>`_.



Getting Started
---------------

Have a look at the :doc:`Getting Started <examples/getting_started/install>` page to train your
first RL agent.


.. hidden tocs .....................................................................................

.. toctree::
    :caption: Examples
    :maxdepth: 1
    :hidden:

    examples/getting_started/index
    examples/stubs/index
    examples/linear_regression/index
    examples/atari/index
    examples/cartpole/index
    examples/frozen_lake/index
    examples/pendulum/index


.. toctree::
    :caption: Function Approximators
    :maxdepth: 1
    :hidden:

    coax/value_functions
    coax/policies
    coax/proba_dists

.. toctree::
    :caption: Update Strategies
    :maxdepth: 1
    :hidden:

    coax/td_learning
    coax/policy_objectives
    coax/policy_regularizers
    coax/reward_tracing
    coax/experience_replay
    coax/value_losses
    coax/value_transforms

.. toctree::
    :caption: Utilities
    :maxdepth: 1
    :hidden:

    coax/decorators
    coax/wrappers
    coax/utils
    coax/envs

.. toctree::
    :caption: Editorial
    :maxdepth: 1
    :hidden:

    Blog Post <blogpost>
    genindex
    release_notes

