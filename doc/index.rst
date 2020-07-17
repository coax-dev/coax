coax
====

*Plug-n-play Reinforcement Learning in Python with OpenAI Gym and Google JAX*


.. image:: /_static/img/cartpole.gif
    :alt: Cartpole-v0 Environment
    :align: center


Create simple, reproducible RL solutions with OpenAI gym environments and JAX-based function
approximators.

.. hidden tocs .....................................................................................

.. toctree::
    :caption: Examples
    :maxdepth: 1
    :hidden:

    examples/prerequisites/index
    examples/stubs/index
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
    coax/value_losses
    coax/value_transforms
    coax/experience_caching
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

    install
    design
    genindex
    release_notes

....................................................................................................


Install
-------

Install using pip:

.. code:: bash

    $ pip install coax

or install from a fresh clone:

.. code:: bash

    $ git clone https://github.com/microsoft/coax.git coax
    $ pip install -e ./coax


N.B. **coax** requires the `JAX <https://jax.readthedocs.io>`_ python package, but it doesn't have
an explicit dependence on it. The reason is that your version of ``jax`` and ``jaxlib`` will depend
on your CUDA version. Check out the :doc:`install` to install ``jax``.


Example
-------

The easiest way to get started is to read through some examples.

Here's one of the examples from the notebooks, in which we solve the ``CartPole-v0`` environment
with the SARSA algorithm, using a simple linear function approximator for our Q-function:


.. literalinclude:: examples/cartpole/sarsa.py
