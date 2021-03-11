REINFORCE
=========

In REINFORCE we don't rely on an external critic. Instead, we use the full Monte-Carlo return as our
feedback signal.

The parametrized policy :math:`\pi_\theta(a|s)` is updated using the following
policy gradients:

.. math::

    g(\theta;S_t,A_t)\ =\ G_t\,\nabla_\theta \log\pi_\theta(A_t|S_t)

where :math:`G_t` is the Monte-Carlo sampled return

.. math::

    G_t\ =\ R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots

The sum is taken over all rewards up to the terminal state.

For more details, see section 13.3 of `Sutton & Barto
<http://incompleteideas.net/book/the-book-2nd.html>`_.


----

:download:`reinforce.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/stubs/reinforce.ipynb

.. literalinclude:: reinforce.py
