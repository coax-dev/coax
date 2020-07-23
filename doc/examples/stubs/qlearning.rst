Q-Learning
==========

Q-learning is a very popular value-based control method. It computes the :math:`n`-step bootstrapped
target as if it were evaluating a greedy policy, i.e.

.. math::

    G^{(n)}_t\ =\ R^{(n)}_t + I^{(n)}_t\,\max_a q(S_{t+n}, a)

where

.. math::

    R^{(n)}_t\ =\ \sum_{k=0}^{n-1}\gamma^kR_{t+k}\ , \qquad
    I^{(n)}_t\ =\ \left\{\begin{matrix}
        0           & \text{if $S_{t+n}$ is a terminal state} \\
        \gamma^n   & \text{otherwise}
    \end{matrix}\right.

For more details, see section 6.5 of `Sutton & Barto
<http://incompleteideas.net/book/the-book-2nd.html>`_. For the **coax** implementation, have a look
at :class:`coax.td_learning.QLearning`.


----

:download:`qlearning.py`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/microsoft/coax/blob/master/doc/_notebooks/stubs/qlearning.ipynb

.. literalinclude:: qlearning.py
