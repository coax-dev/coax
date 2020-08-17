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

import gym
import jax
import jax.numpy as jnp
import numpy as onp
from scipy.linalg import pascal

from .._base.errors import NumpyArrayCheckError


__all__ = (
    'argmax',
    'argmin',
    'batch_to_single',
    'check_array',
    'clipped_logit',
    'diff_transform_matrix',
    'double_relu',
    'get_grads_diagnostics',
    'get_magnitude_quantiles',
    'idx',
    'isscalar',
    'merge_dicts',
    'single_to_batch',
    'safe_sample',
    'tree_ravel',
)


def argmax(rng, arr, axis=-1):
    r"""

    This is a little hack to ensure that argmax breaks ties randomly, which is
    something that :func:`numpy.argmax` doesn't do.

    Parameters
    ----------
    rng : jax.random.PRNGKey

        A pseudo-random number generator key.

    arr : array_like

        Input array.

    axis : int, optional

        By default, the index is into the flattened array, otherwise
        along the specified axis.


    Returns
    -------
    index_array : ndarray of ints

        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    """
    if not isinstance(arr, jnp.ndarray):
        arr = jnp.asarray(arr)
    candidates = arr == jnp.max(arr, axis=axis, keepdims=True)
    logits = (2 * candidates - 1) * 50.  # log(max_float32) == 88.72284
    logits = jnp.moveaxis(logits, axis, -1)
    return jax.random.categorical(rng, logits)


def argmin(rng, arr, axis=-1):
    r"""

    This is a little hack to ensure that argmin breaks ties randomly, which is
    something that :func:`numpy.argmin` doesn't do.

    *Note: random tie breaking is only done for 1d arrays; for multidimensional
    inputs, we fall back to the numpy version.*

    Parameters
    ----------
    rng : jax.random.PRNGKey

        A pseudo-random number generator key.

    arr : array_like

        Input array.

    axis : int, optional

        By default, the index is into the flattened array, otherwise
        along the specified axis.

    Returns
    -------
    index_array : ndarray of ints

        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    """
    return argmax(rng, -arr, axis=axis)


def batch_to_single(pytree, index=0):
    r"""

    Extract a single instance from a :doc:`pytree <pytrees>` of array batches.

    This just does an ``leaf[index]`` on all leaf nodes of the :doc:`pytree
    <pytrees>`.

    Parameters
    ----------
    pytree_batch : pytree with ndarray leaves

        A pytree representing a batch.

    Returns
    -------
    pytree_single : pytree with ndarray leaves

        A pytree representing e.g. a single state observation.

    """
    return jax.tree_map(lambda arr: arr[index], pytree)


def check_array(arr, ndim=None, ndim_min=None, ndim_max=None, dtype=None, shape=None, axis_size=None, axis=None):  # noqa: E501
    """

    This helper function is mostly for internal use. It is used to check a few
    common properties of a numpy array.

    Raises
    ------
    NumpyArrayCheckError

        If one of the checks fails, it raises a :class:`NumpyArrayCheckError`.

    """

    if not isinstance(arr, jnp.ndarray):
        raise NumpyArrayCheckError(
            "expected input to be a numpy array, got type: {}"
            .format(type(arr)))

    check = ndim is not None
    ndims = [ndim] if not isinstance(ndim, (list, tuple, set)) else ndim
    if check and arr.ndim not in ndims:
        raise NumpyArrayCheckError(
            "expected input with ndim(s) {}, got ndim: {}"
            .format(ndim, arr.ndim))

    check = ndim_min is not None
    if check and arr.ndim < ndim_min:
        raise NumpyArrayCheckError(
            "expected input with ndim at least {}, got ndim: {}"
            .format(ndim_min, arr.ndim))

    check = ndim_max is not None
    if check and arr.ndim > ndim_max:
        raise NumpyArrayCheckError(
            "expected input with ndim at most {}, got ndim: {}"
            .format(ndim_max, arr.ndim))

    check = dtype is not None
    dtypes = [dtype] if not isinstance(dtype, (list, tuple, set)) else dtype
    if check and arr.dtype not in dtypes:
        raise NumpyArrayCheckError(
            "expected input with dtype(s) {}, got dtype: {}"
            .format(dtype, arr.dtype))

    check = shape is not None
    if check and arr.shape != shape:
        raise NumpyArrayCheckError(
            "expected input with shape {}, got shape: {}"
            .format(shape, arr.shape))

    check = axis_size is not None and axis is not None
    sizes = [axis_size] if not isinstance(axis_size, (list, tuple, set)) else axis_size  # noqa: E501
    if check and arr.shape[axis] not in sizes:
        raise NumpyArrayCheckError(
            "expected input with size(s) {} along axis {}, got shape: {}"
            .format(axis_size, axis, arr.shape))


def clipped_logit(x, epsilon=1e-15):
    r"""

    A safe implementation of the logit function
    :math:`x\mapsto\log(x/(1-x))`. It clips the arguments of the log function
    from below so as to avoid evaluating it at 0:

    .. math::

        \text{logit}_\epsilon(x)\ =\
            \log(\max(\epsilon, x)) - \log(\max(\epsilon, 1 - x))

    Parameters
    ----------
    x : ndarray

        Input numpy array whose entries lie on the unit interval,
        :math:`x_i\in [0, 1]`.

    epsilon : float, optional

        The small number with which to clip the arguments of the logarithm from
        below.

    Returns
    -------
    z : ndarray, dtype: float, shape: same as input

        The output logits whose entries lie on the real line,
        :math:`z_i\in\mathbb{R}`.

    """
    if jax.api._jit_is_disabled():
        assert jnp.any(x > 0) and jnp.any(x < 1), "values do not lie on the unit interval"
    return jnp.log(
        jnp.maximum(epsilon, x)) - jnp.log(jnp.maximum(epsilon, 1 - x))


def diff_transform_matrix(num_frames, dtype='float32'):
    r"""
    A helper function that implements discrete differentiation for stacked
    state observations.

    Let's say we have a feature vector :math:`X` consisting of four stacked
    frames, i.e. the shape would be: ``[batch_size, height, width, 4]``.

    The corresponding diff-transform matrix with ``num_frames=4`` is a
    :math:`4\times 4` matrix given by:

    .. math::

        M_\text{diff}^{(4)}\ =\ \begin{pmatrix}
            -1 &  0 &  0 & 0 \\
             3 &  1 &  0 & 0 \\
            -3 & -2 & -1 & 0 \\
             1 &  1 &  1 & 1
        \end{pmatrix}

    such that the diff-transformed feature vector is readily computed as:

    .. math::

        X_\text{diff}\ =\ X\, M_\text{diff}^{(4)}

    The diff-transformation preserves the shape, but it reorganizes the frames
    in such a way that they look more like canonical variables. You can think
    of :math:`X_\text{diff}` as the stacked variables :math:`x`,
    :math:`\dot{x}`, :math:`\ddot{x}`, etc. (in reverse order). These
    represent the position, velocity, acceleration, etc. of pixels in a single
    frame.

    Parameters
    ----------
    num_frames : positive int

        The number of stacked frames in the original :math:`X`.

    dtype : keras dtype, optional

        The output data type.

    Returns
    -------
    M : 2d-Tensor, shape: [num_frames, num_frames]

        A square matrix that is intended to be multiplied from the left, e.g.
        ``X_diff = K.dot(X_orig, M)``, where we assume that the frames are
        stacked in ``axis=-1`` of ``X_orig``, in chronological order.

    """
    assert isinstance(num_frames, int) and num_frames >= 1
    s = jnp.diag(jnp.power(-1, jnp.arange(num_frames)))  # alternating sign
    m = s.dot(pascal(num_frames, kind='upper'))[::-1, ::-1]
    return m.astype(dtype)


def double_relu(arr):
    r"""

    A double-ReLU, whose output is the concatenated result of :data:`-relu(-arr) <jax.nn.relu>` and
    :data:`relu(arr) <jax.nn.relu>`.

    This activation function has the advantage that no signal is lost between layers.

    Parameters
    ----------
    arr : ndarray

        The input array, e.g. activations.

    Returns
    -------
    doubled_arr

        The output array, e.g. input for next layer.

    Examples
    --------

    >>> import coax
    >>> import jax.numpy as jnp
    >>> x = jnp.array([[-11, -8],
    ...                [ 17,  5],
    ...                [-13,  7],
    ...                [ 19, -3]])
    ...
    >>> coax.utils.double_relu(x)
    DeviceArray([[-11,  -8,   0,   0],
                 [  0,   0,  17,   5],
                 [-13,   0,   0,   7],
                 [  0,  -3,  19,   0]], dtype=int32)

    There are two things we may observe from the above example. The first is that all components
    from the original array are passed on as output. The second thing is that half of the output
    components (along axis=1) are masked out, which means that the doubling of array size doesn't
    result in doubling the amount of "activation" passed on to the next layer. It merely allows for
    the neural net to learn conditional branches in its internal logic.

    """
    return jnp.concatenate((-jax.nn.relu(-arr), jax.nn.relu(arr)), axis=-1)


def _get_leaf_diagnostics(leaf, key_prefix):
    # update this to add more grads diagnostics
    return {
        f'{key_prefix}max': jnp.max(jnp.abs(leaf)),
        f'{key_prefix}norm': jnp.linalg.norm(jnp.ravel(leaf)),
    }


def get_grads_diagnostics(grads, key_prefix='', keep_tree_structure=False):
    r"""

    Given a :doc:`pytree <pytrees>` of grads, return a dict that contains the quantiles of the
    magnitudes of each individual component.

    This is meant to be a high-level diagnostic. It first extracts the leaves of the pytree, then
    flattens each leaf and then it computes the element-wise magnitude. Then, it concatenates all
    magnitudes into one long flat array. The quantiles are computed on this array.

    Parameters
    ----------
    grads : a pytree with ndarray leaves

        The gradients of some loss function with respect to the model parameters (weights).

    key_prefix : str, optional

        The prefix to add the output dict keys.

    keep_tree_structure : bool, optional

        Whether to keep the tree structure, i.e. to compute the grads diagnostics for each
        individual leaf. If ``False`` (default), we only compute the global grads diagnostics.

    Returns
    -------
    grads_diagnotics : dict<str, float>

        A dict with structure ``{name: score}``.

    """
    if keep_tree_structure:
        return jax.tree_map(lambda g: _get_leaf_diagnostics(g, key_prefix), grads)
    return _get_leaf_diagnostics(tree_ravel(grads), key_prefix)


def get_magnitude_quantiles(pytree, key_prefix=''):
    r"""

    Given a :doc:`pytree <pytrees>`, return a dict that contains the quantiles of the magnitudes of
    each individual component.

    This is meant to be a high-level diagnostic. It first extracts the leaves of the pytree, then
    flattens each leaf and then it computes the element-wise magnitude. Then, it concatenates all
    magnitudes into one long flat array. The quantiles are computed on this array.

    Parameters
    ----------
    pytree : a pytree with ndarray leaves

        A typical example is a pytree of model params (weights) or gradients with respect to such
        model params.

    key_prefix : str, optional

        The prefix to add the output dict keys.

    Returns
    -------
    magnitude_quantiles : dict

        A dict with keys: ``['min', 'p25', 'p50', 'p75', 'max']``. The values of the dict are
        non-negative floats that represent the magnitude quantiles.

    """
    quantiles = jnp.quantile(jnp.abs(tree_ravel(pytree)), jnp.array([0, 0.25, 0.5, 0.75, 1]))
    quantile_names = (f'{key_prefix}{k}' for k in ('min', 'p25', 'p50', 'p75', 'max'))
    return dict(zip(quantile_names, quantiles))


def idx(arr, axis=0):
    r"""
    Given a numpy array, return its corresponding integer index array.

    Parameters
    ----------
    arr : array

        Input array.

    axis : int, optional

        The axis along which we'd like to get an index.

    Returns
    -------
    index : 1d array, shape: arr.shape[axis]

        An index array `[0, 1, 2, ...]`.

    """
    check_array(arr, ndim_min=1)
    return jnp.arange(arr.shape[axis])


def isscalar(num):
    r"""

    This helper uses a slightly looser definition of scalar compared to
    :func:`numpy.isscalar` (and :func:`jax.numpy.isscalar`) in that it also
    considers single-item arrays to be scalaras as well.

    Parameters
    ----------
    num : number or ndarray

        Input array.

    Returns
    -------
    isscalar : bool

        Whether the input number is either a number or a single-item array.

    """
    return jnp.isscalar(num) or (
        isinstance(num, (jnp.ndarray, onp.ndarray)) and jnp.size(num) == 1)


def merge_dicts(*dicts):
    r"""

    Merge dicts into a single dict.

    WARNING: duplicate keys are not resolved.

    Parameters
    ----------
    \*dicts : \*dict

        Multiple dictionaries.

    Returns
    -------
    merged : dict

        A single dictionary.

    """
    merged = {}
    for d in dicts:
        overlap = set(d).intersection(merged)
        if overlap:
            warnings.warn(f"merge_dicts found overlapping keys: {tuple(overlap)}")
        merged.update(d)
    return merged


def _safe_sample(space, rnd):
    if isinstance(space, gym.spaces.Discrete):
        return rnd.randint(space.n)

    if isinstance(space, gym.spaces.MultiDiscrete):
        return onp.asarray([rnd.randint(n) for n in space.nvec])

    if isinstance(space, gym.spaces.MultiBinary):
        return rnd.randint(2, size=space.n)

    if isinstance(space, gym.spaces.Box):
        return onp.clip(rnd.rand(*space.shape), space.low, space.high)

    if isinstance(space, gym.spaces.Tuple):
        return tuple(_safe_sample(sp, rnd) for sp in space.spaces)

    if isinstance(space, gym.spaces.Dict):
        return {k: _safe_sample(space.spaces[k], rnd) for k in sorted(space.spaces)}

    # fallback for non-supported spaces
    return space.sample()


def safe_sample(space, seed=None):
    r"""

    Safely sample from a gym-style space.

    Parameters
    ----------
    space : gym.Space

        A gym-style space.

    seed : int, optional

        The seed for the pseudo-random number generator.

    Returns
    -------
    sample

        An single sample from of the given ``space``.

    """
    if not isinstance(space, gym.Space):
        raise TypeError("space must be derived from gym.Space")

    rnd = seed if isinstance(seed, onp.random.RandomState) else onp.random.RandomState(seed)
    return _safe_sample(space, rnd)


def single_to_batch(pytree):
    r"""

    Take a single instance and turn it into a batch of size 1.

    This just does an ``np.expand_dims(leaf, axis=0)`` on all leaf nodes of the
    :doc:`pytree <pytrees>`.

    Parameters
    ----------
    pytree_single : pytree with ndarray leaves

        A pytree representing e.g. a single state observation.

    Returns
    -------
    pytree_batch : pytree with ndarray leaves

        A pytree representing a batch with ``batch_size=1``.

    """
    return jax.tree_map(lambda arr: jnp.expand_dims(arr, axis=0), pytree)


def tree_ravel(pytree):
    r"""

    Flatten and concatenate all leaves into a single flat ndarray.

    Parameters
    ----------
    pytree : a pytree with ndarray leaves

        A typical example is a pytree of model parameters (weights) or gradients with respect to
        such model params.

    Returns
    -------
    arr : ndarray with ndim=1

        A single flat array.

    """
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_leaves(pytree)])
