SARSA
=====

SARSA is probably the simplest value-based control method. The :math:`n`-step bootstrapped target is
constructed as:

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,q(S_{t+n}, A_{t+n})

where :math:`A_{t+n}` is sampled from experience and

.. math::

    R^{(n)}_t\ =\ \sum_{k=0}^{n-1}\gamma^kR_{t+k}\ , \qquad
    I^{(n)}_t\ =\ \left\{\begin{matrix}
        0           & \text{if $S_{t+n}$ is a terminal state} \\
        \gamma^n   & \text{otherwise}
    \end{matrix}\right.

For more details, see section 6.4 of `Sutton & Barto
<http://incompleteideas.net/book/the-book-2nd.html>`_. For the **coax** implementation, have a look
at :class:`coax.td_learning.Sarsa`.


----

:download:`sarsa.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/main/doc/_notebooks/stubs/sarsa.ipynb

.. literalinclude:: sarsa.py
