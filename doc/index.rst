coax
====

*Plug-n-play Reinforcement Learning in Python with OpenAI Gym and JAX*


.. image:: /_static/img/cartpole.gif
    :alt: Cartpole-v0 Environment
    :align: center


Create simple, reproducible RL solutions with OpenAI gym environments and JAX-based function
approximators.

Install
-------

Install using pip:

.. code:: bash

    $ pip install coax

or install from a fresh clone:

.. code:: bash

    $ git clone https://github.com/microsoft/coax.git
    $ pip install -e ./coax


.. note::

    **coax** requires the JAX python package, but it doesn't have an explicit dependence on it. The
    reason is that your version of ``jax`` and ``jaxlib`` will depend on your CUDA version. Check
    out the :doc:`examples/getting_started/install` to install ``jax``.


Getting Started
---------------

Have a look at the :doc:`Getting Started <examples/getting_started/index>` page to train your first
RL agent.


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

    coax/func_approx
    coax/policies
    coax/value_functions
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
    coax/planning

.. toctree::
    :caption: Utilities
    :maxdepth: 1
    :hidden:

    coax/wrappers
    coax/utils
    coax/envs

.. toctree::
    :caption: Editorial
    :maxdepth: 1
    :hidden:

    design
    genindex
    release_notes

