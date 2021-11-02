import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp

__all__ = (
    'quantiles',
    'quantiles_uniform',
    'quantile_cos_embedding'
)


def quantiles_uniform(rng, batch_size, num_quantiles=32):
    """
    Generate :code:`batch_size` quantile fractions that split the interval :math:`[0, 1]`
    into :code:`num_quantiles` uniformly distributed fractions.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        A pseudo-random number generator key.
    batch_size : int
        The batch size for which the quantile fractions should be generated.
    num_quantiles : int, optional
        The number of quantile fractions. By default 32.

    Returns
    -------
    quantile_fractions : ndarray
        Array of quantile fractions.
    """
    rngs = hk.PRNGSequence(rng)
    quantile_fractions = jax.random.uniform(next(rngs), shape=(batch_size, num_quantiles))
    quantile_fraction_differences = quantile_fractions / \
        jnp.sum(quantile_fractions, axis=-1, keepdims=True)
    quantile_fractions = jnp.cumsum(quantile_fraction_differences, axis=-1)
    return quantile_fractions


def quantiles(batch_size, num_quantiles=200):
    r"""
    Generate :code:`batch_size` quantile fractions that split the interval :math:`[0, 1]`
    into :code:`num_quantiles` equally spaced fractions.

    Parameters
    ----------
    batch_size : int
        The batch size for which the quantile fractions should be generated.
    num_quantiles : int, optional
        The number of quantile fractions. By default 200.

    Returns
    -------
    quantile_fractions : ndarray

        Array of quantile fractions.
    """
    quantile_fractions = jnp.arange(num_quantiles, dtype=jnp.float32) / num_quantiles
    quantile_fractions = jnp.tile(quantile_fractions[None, :], [batch_size, 1])
    return quantile_fractions


def quantile_cos_embedding(quantile_fractions, n=64):
    r"""
    Embed the given quantile fractions :math:`\tau` in an `n` dimensional space
    using cosine basis functions.

    .. math::

        \phi(\tau) = \cos(\tau i \pi) \qquad 0 \leq i \lt n


    Parameters
    ----------
    quantile_fractions : ndarray

        Array of quantile fractions :math:`\tau` to be embedded.

    n : int

        The dimensionality of the embedding. By default 64.

    Returns
    -------
    quantile_embs : ndarray

        Array of quantile embeddings with shape `(quantile_fractions.shape[0], n)`.
    """
    quantile_fractions = jnp.tile(quantile_fractions[..., None],
                                  [1] * quantile_fractions.ndim + [n])
    quantiles_emb = (
        jnp.arange(1, n + 1, 1)
        * onp.pi
        * quantile_fractions)
    quantiles_emb = jnp.cos(quantiles_emb)
    return quantiles_emb
