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

import jax
import jax.numpy as jnp


__all__ = (
    'mse',
    'huber',
    'logloss',
    'logloss_sign',
)


def mse(y_true, y_pred, w=None):
    r"""

    Ordinary mean-squared error loss function.

    .. math::

        L\ =\ \frac12(\hat{y} - y)^2

    .. image:: /_static/img/mse.svg
        :alt: Mean-Squared Error loss
        :width: 320px
        :align: center

    Parameters
    ----------
    y_true : ndarray

        The target :math:`y\in\mathbb{R}`.

    y_pred : ndarray

        The predicted output :math:`\hat{y}\in\mathbb{R}`.

    w : ndarray, optional

        Sample weights.

    Returns
    -------
    loss : scalar ndarray

        The loss averaged over the batch.

    """
    loss = 0.5 * jnp.square(y_pred - y_true)
    return _mean_with_weights(loss, w)


def huber(y_true, y_pred, w=None, delta=1.0):
    r"""

    `Huber <https://en.wikipedia.org/wiki/Huber_loss>`_ loss function.

    .. math::

        L\ =\ \left\{\begin{matrix}
                (\hat{y} - y)^2
                    &\quad:\ |\hat{y} - y|\leq\delta \\
                \delta\,|\hat{y} - y| - \frac{\delta^2}{2}
                    &\quad:\ |\hat{y} - y| > \delta
            \end{matrix}\right.

    .. image:: /_static/img/huber.svg
        :alt: Huber loss
        :width: 320px
        :align: center

    Parameters
    ----------
    y_true : ndarray

        The target :math:`y\in\mathbb{R}`.

    y_pred : ndarray

        The predicted output :math:`\hat{y}\in\mathbb{R}`.

    w : ndarray, optional

        Sample weights.

    delta : float, optional

        The scale of the quadratic-to-linear transition.

    Returns
    -------
    loss : scalar ndarray

        The loss averaged over the batch.

    """
    err = jnp.abs(y_pred - y_true)
    err_clipped = jnp.minimum(err, delta)
    loss = 0.5 * jnp.square(err_clipped) + delta * (err - err_clipped)
    return _mean_with_weights(loss, w)


def logloss(y_true, y_pred, w=None):
    r"""

    Logistic loss function for binary classification, `y_true` =
    :math:`y\in\{0,1\}` and the model output is a probability `y_pred` =
    :math:`\hat{y}\in[0,1]`:

    .. math::

        L\ =\ -y\log(\hat{y}) - (1 - y)\log(1 - \hat{y})

    Parameters
    ----------
    y_true : ndarray

        The binary target, encoded as :math:`y\in\{0,1\}`.

    y_pred : (ndarray of) float

        The predicted output, represented by a probablity
        :math:`\hat{y}\in[0,1]`.

    w : ndarray, optional

        Sample weights.

    Returns
    -------
    loss : scalar ndarray

        The loss averaged over the batch.

    """
    loss = -y_true * jnp.log(y_pred) - (1. - y_true) * jnp.log(1. - y_pred)
    return _mean_with_weights(loss, w)


def logloss_sign(y_true_sign, logits, w=None):
    r"""

    Logistic loss function specific to the case in which the target is a sign
    :math:`y\in\{-1,1\}` and the model output is a logit
    :math:`\hat{z}\in\mathbb{R}`.

    .. math::

        L\ =\ \log(1 + \exp(-y\,\hat{z}))

    This version tends to be more numerically stable than the generic
    implementation, because it avoids having to map the predicted logit to a
    probability.

    Parameters
    ----------
    y_true_sign : ndarray

        The binary target, encoded as :math:`y=\pm1`.

    logits : ndarray

        The predicted output, represented by a logit
        :math:`\hat{z}\in\mathbb{R}`.

    w : ndarray, optional

        Sample weights.

    Returns
    -------
    loss : scalar ndarray

        The loss averaged over the batch.

    """
    loss = jnp.log(1.0 + jnp.exp(-y_true_sign * logits))
    return _mean_with_weights(loss, w)


def _mean_with_weights(loss, w):
    if w is not None:
        assert w.ndim == 1
        assert loss.ndim >= 1
        assert loss.shape[0] == w.shape[0]
        loss = jax.vmap(jnp.multiply)(w, loss)
    return jnp.mean(loss)
