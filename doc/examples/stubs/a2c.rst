Advantage Actor-Critic (A2C)
============================

Advantage Actor-Critic (A2C) is probably the simplest actor-critic. Instead of using a q-function as
its critic, it used the fact that the advantage function can be intepreted as the expectation value
of the TD error. To see this, use the definition of the q-function to express the advantage function
as:

.. math::

    \mathcal{A}(s,a)\ =\ q(s,a) - v(s)\ =\
        \mathbb{E}_t \left\{G_t - v(s)\,|\, S_t=s, A_t=a\right\}

Then, we replace :math:`G_t` by our bootstrapped estimate:

.. math::

    G_t\ \approx\ G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,v(S_{t+n})

where

.. math::

    R^{(n)}_t\ =\ \sum_{k=0}^{n-1}\gamma^kR_{t+k}\ , \qquad
    I^{(n)}_t\ =\ \left\{\begin{matrix}
        0           & \text{if $S_{t+n}$ is a terminal state} \\
        \gamma^n   & \text{otherwise}
    \end{matrix}\right.

The parametrized policy :math:`\pi_\theta(a|s)` is updated using the following policy gradients:

.. math::

    g(\theta;S_t,A_t)\
        &=\ \mathcal{A}(S_t,A_t)\,\nabla_\theta \log\pi_\theta(A_t|S_t) \\
        &\approx\ \left(G^{(n)}_t - v(S_t)\right)\,
            \nabla_\theta \log\pi_\theta(A_t|S_t)


The prefactor :math:`G^{(n)}_t - v(S_t)` is known as the TD error.

For more details, see section 13.5 of `Sutton & Barto
<http://incompleteideas.net/book/the-book-2nd.html>`_.


----

:download:`a2c.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/main/doc/_notebooks/stubs/a2c.ipynb

.. literalinclude:: a2c.py
