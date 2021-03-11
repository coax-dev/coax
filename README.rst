|tests| |pypi| |docs| |License|


coax
====

Plug-n-Play Reinforcement Learning in Python with `OpenAI Gym <https://gym.openai.com>`_ and
`JAX <https://jax.readthedocs.io>`_

|readthedocs|

For the full documentation, including many examples, go to https://coax.readthedocs.io/


Install
-------

**coax** is built on top of JAX, but it doesn't have an explicit dependence on the ``jax`` python
package. The reason is that your version of ``jaxlib`` will depend on your CUDA version. To install
without CUDA, simply run:

.. code-block::

    $ pip install jaxlib jax coax --upgrade


If you do require CUDA support, please check out the
`Installation Guide <https://coax.readthedocs.io/examples/getting_started/install.html>`_.


Getting Started
---------------

Have a look at the
`Getting Started <https://coax.readthedocs.io/examples/getting_started/prereq_jax.html>`_ page to
train your first RL agent.


....................................................................................................

.. |readthedocs| image:: https://raw.githubusercontent.com/coax-dev/coax/main/doc/_static/img/readthedocs.gif
    :target: https://coax.readthedocs.io/
    :width: 400
    :alt: readthedocs

.. |tests| image:: https://github.com/coax-dev/coax/workflows/tests/badge.svg
    :target: https://github.com/coax-dev/coax/actions?query=workflow%3Atests
    :alt: tests badge

.. |pypi| image:: https://img.shields.io/pypi/v/coax
    :target: https://pypi.org/project/coax
    :alt: pypi badge

.. |docs| image:: https://readthedocs.org/projects/coax/badge/?version=latest
    :target: https://coax.readthedocs.io
    :alt: docs badge

.. |license| image:: https://img.shields.io/github/license/coax-dev/coax
    :target: https://github.com/coax-dev/coax/blob/main/LICENSE
    :alt: license badge
