import pytest
import jax
import haiku as hk

from ._array import unvectorize


@pytest.fixture
def rngs():
    return hk.PRNGSequence(42)


@pytest.fixture
def x_batch():
    rng = jax.random.PRNGKey(13)
    return jax.random.normal(rng, shape=(7, 11))


@pytest.fixture
def x_single():
    rng = jax.random.PRNGKey(17)
    return jax.random.normal(rng, shape=(11,))


def test_unvectorize_single_output(rngs, x_batch, x_single):
    def f_batch(X):
        return hk.Linear(11)(X)

    init, f_batch = hk.transform(f_batch)
    params = init(next(rngs), x_batch)
    y_batch = f_batch(params, next(rngs), x_batch)
    assert y_batch.shape == (7, 11)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0), out_axes=0)
    y_single = f_single(params, next(rngs), x_single)
    assert y_single.shape == (11,)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0), out_axes=(0,))
    msg = r"out_axes must be an int for functions with a single output; got: out_axes=\(0,\)"
    with pytest.raises(TypeError, match=msg):
        f_single(params, next(rngs), x_single)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0, 0), out_axes=(0,))
    msg = r"number of in_axes must match the number of function inputs"
    with pytest.raises(ValueError, match=msg):
        f_single(params, next(rngs), x_single)


def test_unvectorize_multi_output(rngs, x_batch, x_single):
    def f_batch(X):
        return hk.Linear(11)(X), hk.Linear(13)(X)

    init, f_batch = hk.transform(f_batch)
    params = init(next(rngs), x_batch)
    y_batch = f_batch(params, next(rngs), x_batch)
    assert y_batch[0].shape == (7, 11)
    assert y_batch[1].shape == (7, 13)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0), out_axes=0)
    y_single = f_single(params, next(rngs), x_single)
    assert y_single[0].shape == (11,)
    assert y_single[1].shape == (13,)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0), out_axes=(0, None))
    y_single = f_single(params, next(rngs), x_single)
    assert y_single[0].shape == (11,)
    assert y_single[1].shape == (1, 13)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0), out_axes=None)
    y_single = f_single(params, next(rngs), x_single)
    assert y_single[0].shape == (1, 11,)
    assert y_single[1].shape == (1, 13)

    f_single = unvectorize(f_batch, in_axes=(None, None, 0), out_axes=(0,))
    msg = r"number of out_axes must match the number of function outputs"
    with pytest.raises(ValueError, match=msg):
        f_single(params, next(rngs), x_single)
