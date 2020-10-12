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


Contributing
------------

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a
CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the
`Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_. For more
information see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or
contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_ with any additional questions or
comments.

....................................................................................................

.. |readthedocs| image:: https://raw.githubusercontent.com/microsoft/coax/main/doc/_static/img/readthedocs.gif
    :target: https://coax.readthedocs.io/
    :width: 400
    :alt: readthedocs

.. |tests| image:: https://github.com/microsoft/coax/workflows/tests/badge.svg
    :target: https://github.com/microsoft/coax/actions?query=workflow%3Atests
    :alt: tests badge

.. |pypi| image:: https://img.shields.io/pypi/v/coax
    :target: https://pypi.org/project/coax
    :alt: pypi badge

.. |docs| image:: https://readthedocs.org/projects/coax/badge/?version=latest
    :target: https://coax.readthedocs.io
    :alt: docs badge

.. |license| image:: https://img.shields.io/github/license/microsoft/coax
    :target: https://github.com/microsoft/coax/blob/main/LICENSE
    :alt: license badge
