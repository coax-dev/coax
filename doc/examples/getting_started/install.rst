**Steps:** [ *install* | :doc:`jax <prereq_jax>` | :doc:`haiku <prereq_haiku>` | :doc:`q-learning <first_agent>` | :doc:`dqn <second_agent>` | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

Installation Guide
==================


In this guide you'll find out how to install the ``coax`` package.

A guide to installing `JAX <https://jax.readthedocs.io>`_ is included as well.
Even though ``coax`` requires the ``jax``, it doesn't have an explicit
dependence on it. The reason is that your version of ``jax`` and ``jaxlib``
will depend on your CUDA version.


Install jax
-----------

There are *three ways* to install JAX. If you're not sure which one to pick, just go for
**Option 1**.


.. tabs::

    .. tab:: Option 1: Default

        To install the default version of ``jax`` and ``jaxlib`` you can run

        .. code:: bash

            $ pip install -U jax jaxlib


    .. tab:: Option 2: GPU support

        To install a specific version that will be compatible with your CUDA
        installation, have a look at the versions listed at:

            https://storage.googleapis.com/jax-releases/

        To install a specific version, note down (or copy-paste) the correct version
        numbers. For instance, at the time of writing, I'm running CPython version 3.6
        (``cp36``) and my CUDA version is 10.2 (``cuda102``) and the latest version
        available is 0.1.41 (``jaxlib-0.1.41``), so I download and install this
        version:

        .. code:: bash

            $ wget https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.41-cp36-none-linux_x86_64.whl
            $ pip install jaxlib-0.1.41-cp36-none-linux_x86_64.whl
            $ pip install -U jax


        N.B. ``jax`` will tell you if it requires a more recent version of
        ``jaxlib``.


    .. tab:: Option 3: Build from source

        To build ``jaxlib`` from source, you can follow the developer guide from the
        JAX docs:

            https://jax.readthedocs.io/en/latest/developer.html#building-from-source


Install coax
------------

Install using pip:

.. code:: bash

    $ pip install coax

or install from a fresh clone:

.. code:: bash

    $ git clone https://github.com/microsoft/coax.git
    $ pip install -e ./coax
