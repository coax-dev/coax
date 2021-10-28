import jax
import jax.numpy as jnp
import haiku as hk


def quantile_func_iqn(S, rng, is_training, A=None, num_quantiles=32):
    rngs = hk.PRNGSequence(rng)
    batch_size = jax.tree_leaves(S)[0].shape[0]
    quantiles = [jax.random.uniform(next(rngs), shape=(1, num_quantiles))
                 for _ in range(batch_size)]
    quantiles = [quantiles_fractions /
                 jnp.sum(quantiles_fractions, axis=-1, keepdims=True)
                 for quantiles_fractions in quantiles]
    quantiles = [jnp.cumsum(quantiles_fractions, axis=-1) for quantiles_fractions in quantiles]
    quantiles = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *quantiles)
    return quantiles


def quantile_func_qrdqn(S, rng, is_training, A=None, num_quantiles=200):
    batch_size = jax.tree_leaves(S)[0].shape[0]
    quantiles = [jnp.arange(num_quantiles, dtype=jnp.float32) /
                 num_quantiles for _ in range(batch_size)]
    quantiles = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *quantiles)
    return quantiles
