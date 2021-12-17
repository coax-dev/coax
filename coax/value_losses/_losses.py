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


def quantile_huber(y_true, y_pred, quantiles, w=None, delta=1.0):
    r"""

    `Quantile Huber <https://arxiv.org/abs/1806.06923>`_ loss function.

    .. math::

        \delta_{ij} &= y_j - \hat{y}_i\\
        \rho^\kappa_\tau(\delta_{ij}) &= |\tau - \mathbb{I}{\{ \delta_{ij} < 0 \}}| \
            \frac{\mathcal{L}_\kappa(\delta_{ij})}{\kappa},\ \quad \text{with}\\
        \mathcal{L}_\kappa(\delta_{ij}) &= \begin{cases}
            \frac{1}{2} \delta_{ij}^2,\quad \ &\text{if } |\delta_{ij}| \le \kappa\\
            \kappa (|\delta_{ij}| - \frac{1}{2}\kappa),\quad \ &\text{otherwise}
        \end{cases}

    Parameters
    ----------
    y_true : ndarray

        The target :math:`y\in\mathbb{R}^{2}`.

    y_pred : ndarray

        The predicted output :math:`\hat{y}\in\mathbb{R}^{2}`.

    quantiles : ndarray

        The quantiles of the prediction :math:`\tau\in\mathbb{R}^{2}`.

    w : ndarray, optional

        Sample weights.

    delta : float, optional

        The scale of the quadratic-to-linear transition.

    Returns
    -------
    loss : scalar ndarray

        The loss averaged over the batch.

    """
    y_pred = y_pred[..., None]
    y_true = y_true[..., None, :]
    quantiles = quantiles[..., None]
    td_error = y_true - y_pred
    td_error_abs = jnp.abs(td_error)
    err_clipped = jnp.minimum(td_error_abs, delta)
    elementwise_huber_loss = 0.5 * jnp.square(err_clipped) + delta * (td_error_abs - err_clipped)
    elementwise_quantile_huber_loss = jnp.abs(
        quantiles - (td_error < 0)) * elementwise_huber_loss / delta
    quantile_huber_loss = elementwise_quantile_huber_loss.sum(axis=-1)
    return _mean_with_weights(quantile_huber_loss, w=w)
