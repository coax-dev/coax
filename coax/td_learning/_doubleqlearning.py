# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

import warnings

import haiku as hk
from gym.spaces import Discrete

from ._base import BaseTDLearningQWithTargetPolicy


class DoubleQLearning(BaseTDLearningQWithTargetPolicy):
    r"""

    TD-learning with `target-style double q-learning <https://arxiv.org/abs/1509.06461>`_ updates,
    in which the target network is only used in selecting the would-be next action. The
    :math:`n`-step bootstrapped target is thus constructed as:

    .. math::

        a_\text{greedy}\ &=\ \arg\max_a q_\text{targ}(S_{t+n}, a) \\
        G^{(n)}_t\ &=\ R^{(n)}_t + I^{(n)}_t\,q(S_{t+n}, a_\text{greedy})

    where

    .. math::

        R^{(n)}_t\ &=\ \sum_{k=0}^{n-1}\gamma^kR_{t+k} \\
        I^{(n)}_t\ &=\ \left\{\begin{matrix}
            0           & \text{if $S_{t+n}$ is a terminal state} \\
            \gamma^n    & \text{otherwise}
        \end{matrix}\right.

    Parameters
    ----------
    q : Q

        The main q-function to update.

    pi_targ : Policy, optional

        The policy that is used for constructing the TD-target. This is ignored if the action space
        is discrete and *required* otherwise.

    q_targ : Q, optional

        The q-function that is used for constructing the TD-target. If this is left unspecified, we
        set ``q_targ = q`` internally.

    optimizer : optax optimizer, optional

        An optax-style optimizer. The default optimizer is :func:`optax.adam(1e-3)
        <optax.adam>`.

    loss_function : callable, optional

        The loss function that will be used to regress to the (bootstrapped) target. The loss
        function is expected to be of the form:

        .. math::

            L(y_\text{true}, y_\text{pred})\in\mathbb{R}

        If left unspecified, this defaults to :func:`coax.value_losses.huber`. Check out the
        :mod:`coax.value_losses` module for other predefined loss functions.

    value_transform : ValueTransform or pair of funcs, optional

        If provided, the returns are transformed as follows:

        .. math::

            G^{(n)}_t\ \mapsto\ f\left(G^{(n)}_t\right)\ =\
                f\left(R^{(n)}_t + I^{(n)}_t\,f^{-1}\left(v(S_{t+n})\right)\right)

        where :math:`f` and :math:`f^{-1}` are given by ``value_transform.transform_func`` and
        ``value_transform.inverse_func``, respectively. See :mod:`coax.td_learning` for examples of
        value-transforms. Note that a ValueTransform is just a glorified pair of functions, i.e.
        passing ``value_transform=(func, inverse_func)`` works just as well.

    """
    def __init__(
            self, q, pi_targ=None, q_targ=None,
            optimizer=None, loss_function=None, value_transform=None):

        super().__init__(
            q=q, pi_targ=pi_targ, q_targ=q_targ, optimizer=optimizer,
            loss_function=loss_function, value_transform=value_transform)

        # consistency checks
        if self.pi_targ is None and not isinstance(self.q.action_space, Discrete):
            raise TypeError("pi_targ must be provided if action space is not discrete")
        if self.pi_targ is not None and isinstance(self.q.action_space, Discrete):
            warnings.warn("pi_targ is ignored, because action space is discrete")

    def target_func(self, target_params, target_state, rng, transition_batch):
        rngs = hk.PRNGSequence(rng)
        Rn, In, S_next = transition_batch[3:6]

        if isinstance(self.q.action_space, Discrete):
            # compute A_next as the argmax over q_targ
            params, state = target_params['q_targ'], target_state['q_targ']
            Q_s_next, _ = self.q_targ.function_type2(params, state, next(rngs), S_next, False)
            assert Q_s_next.ndim == 2, f"bad shape: {Q_s_next.shape}"
            A_next = (Q_s_next == Q_s_next.max(axis=1, keepdims=True)).astype(Q_s_next.dtype)
            A_next /= A_next.sum(axis=1, keepdims=True)  # there may be ties

        else:
            # compute A_next as the mode of pi_targ
            params, state = target_params['pi_targ'], target_state['pi_targ']
            dist_params, _ = self.pi_targ.function(params, state, next(rngs), S_next, False)
            A_next = self.pi_targ.proba_dist.mode(dist_params)  # greedy action

        # evaluate on q (not q_targ)
        params, state = target_params['q'], target_state['q']
        Q_sa_next, _ = self.q.function_type1(params, state, next(rngs), S_next, A_next, False)

        assert Q_sa_next.ndim == 1, f"bad shape: {Q_sa_next.shape}"
        f, f_inv = self.value_transform
        return f(Rn + In * f_inv(Q_sa_next))
