import warnings
from functools import partial

import chex
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk
from scipy.linalg import pascal


__all__ = (
    'StepwiseLinearFunction',
    'argmax',
    'argmin',
    'batch_to_single',
    'check_array',
    'check_preprocessors',
    'chunks_pow2',
    'clipped_logit',
    'default_preprocessor',
    'diff_transform',
    'diff_transform_matrix',
    'double_relu',
    'get_grads_diagnostics',
    'get_magnitude_quantiles',
    'get_transition_batch',
    'idx',
    'isscalar',
    'merge_dicts',
    'single_to_batch',
    'safe_sample',
    'stack_trees',
    'tree_ravel',
    'tree_sample',
    'unvectorize',
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


def check_array(
        arr, ndim=None, ndim_min=None, ndim_max=None,
        dtype=None, shape=None, axis_size=None, axis=None, except_np=False):
    r"""

    This helper function is mostly for internal use. It is used to check a few
    common properties of a numpy array.

    Raises
    ------
    TypeError

        If one of the checks fails.

    """
    if not except_np and not isinstance(arr, jnp.ndarray):
        raise TypeError(f"expected input to be a jnp.ndarray, got type: {type(arr)}")
    if not isinstance(arr, (onp.ndarray, jnp.ndarray)):
        raise TypeError(f"expected input to be an ndarray, got type: {type(arr)}")

    check = ndim is not None
    ndims = [ndim] if not isinstance(ndim, (list, tuple, set)) else ndim
    if check and arr.ndim not in ndims:
        raise TypeError(f"expected input with ndim(s) {ndim}, got ndim: {arr.ndim}")

    check = ndim_min is not None
    if check and arr.ndim < ndim_min:
        raise TypeError(f"expected input with ndim at least {ndim_min}, got ndim: {arr.ndim}")

    check = ndim_max is not None
    if check and arr.ndim > ndim_max:
        raise TypeError(f"expected input with ndim at most {ndim_max}, got ndim: {arr.ndim}")

    check = dtype is not None
    dtypes = [dtype] if not isinstance(dtype, (list, tuple, set)) else dtype
    if check and arr.dtype not in dtypes:
        raise TypeError(f"expected input with dtype(s) {dtype}, got dtype: {arr.dtype}")

    check = shape is not None
    if check and arr.shape != shape:
        raise TypeError(f"expected input with shape {shape}, got shape: {arr.shape}")

    check = axis_size is not None and axis is not None
    sizes = [axis_size] if not isinstance(axis_size, (list, tuple, set)) else axis_size
    if check and arr.shape[axis] not in sizes:
        raise TypeError(
            f"expected input with size(s) {axis_size} along axis {axis}, got shape: {arr.shape}")


def check_preprocessors(space, *preprocessors, num_samples=20, random_seed=None):
    r"""

    Check whether two preprocessors are the same.

    Parameters
    ----------
    space : gym.Space

        The domain of the prepocessors.

    \*preprocessors

        Preprocessor functions, which are functions with input signature: :code:`func(rng: PRNGKey,
        x: Element[space]) -> Any`.

    num_samples : positive int

        The number of samples in which to run checks.

    Returns
    -------
    match : bool

        Whether the preprocessors match.

    """
    if len(preprocessors) < 2:
        raise ValueError("need at least two preprocessors in order to run test")

    def test_leaves(a, b):
        assert type(a) is type(b)
        return onp.testing.assert_allclose(onp.asanyarray(a), onp.asanyarray(b))

    rngs = hk.PRNGSequence(onp.random.RandomState(random_seed).randint(jnp.iinfo('int32').max))
    p0, *ps = preprocessors

    with jax.disable_jit():
        for _ in range(num_samples):
            x = space.sample()
            y0 = p0(next(rngs), x)
            for p in ps:
                y = p(next(rngs), x)
                if jax.tree_structure(y) != jax.tree_structure(y0):
                    return False
                try:
                    jax.tree_multimap(test_leaves, y, y0)
                except AssertionError:
                    return False
    return True


def chunks_pow2(transition_batch):
    r"""

    Split up a :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>`
    into smaller batches with sizes equal to powers of 2. This is useful
    to recude overhead due to repeated JIT compilation due to varying batch sizes.

    Yields
    ------
    chunk : TransitionBatch

        A smaller chunk with batch_size equal to a power of 2.

    """
    def leafslice(start, stop):
        def func(leaf):
            if leaf is None:
                return None
            return leaf[start:stop]
        return func

    binary = bin(transition_batch.batch_size).replace('0b', '')
    start = 0
    for i, b in enumerate(binary, 1):
        if b == '0':
            continue
        stop = start + 2 ** (len(binary) - i)
        yield jax.tree_map(leafslice(start, stop), transition_batch)
        start = stop


def clipped_logit(x, epsilon=1e-15):
    r"""

    A safe implementation of the logit function :math:`x\mapsto\log(x/(1-x))`. It clips the
    arguments of the log function from below so as to avoid evaluating it at 0:

    .. math::

        \text{logit}_\epsilon(x)\ =\
            \log(\max(\epsilon, x)) - \log(\max(\epsilon, 1 - x))

    Parameters
    ----------
    x : ndarray

        Input numpy array whose entries lie on the unit interval, :math:`x_i\in [0, 1]`.

    epsilon : float, optional

        The small number with which to clip the arguments of the logarithm from below.

    Returns
    -------
    z : ndarray, dtype: float, shape: same as input

        The output logits whose entries lie on the real line,
        :math:`z_i\in\mathbb{R}`.

    """
    return jnp.log(jnp.clip(x, epsilon, 1)) - jnp.log(jnp.clip(1 - x, epsilon, 1))


def default_preprocessor(space):
    r"""

    The default preprocessor for a given space.

    Parameters
    ----------
    space : gym.Space

        The domain of the prepocessor.

    Returns
    -------
    preprocessor : Callable[PRGNKey, Element[space], Any]

        The preprocessor function. See :attr:`NormalDist.preprocess_variate
        <coax.proba_dists.NormalDist.preprocess_variate>` for an example.

    """
    if not isinstance(space, gym.Space):
        raise TypeError(f"space must a gym.Space, got: {type(space)}")

    if isinstance(space, gym.spaces.Discrete):
        def func(rng, X):
            X = jnp.asarray(X)
            X = jax.nn.one_hot(X, space.n)     # one-hot encoding
            X = jnp.reshape(X, (-1, space.n))  # ensure batch axis
            return X

    elif isinstance(space, gym.spaces.Box):
        def func(rng, X):
            X = jnp.asarray(X, dtype=space.dtype)   # ensure ndarray
            X = jnp.reshape(X, (-1, *space.shape))  # ensure batch axis
            X = jnp.clip(X, space.low, space.high)  # clip to be safe
            return X

    elif isinstance(space, gym.spaces.MultiDiscrete):
        def func(rng, X):
            rngs = jax.random.split(rng, len(space.nvec))
            chex.assert_rank(X, {1, 2})
            if X.ndim == 1:
                X = jnp.expand_dims(X, axis=0)
            return [
                default_preprocessor(gym.spaces.Discrete(n))(rng, X[:, i])
                for i, (n, rng) in enumerate(zip(space.nvec, rngs))]

    elif isinstance(space, gym.spaces.MultiBinary):
        def func(rng, X):
            X = jnp.asarray(X, dtype=jnp.float32)  # ensure ndarray
            X = jnp.reshape(X, (-1, space.n))      # ensure batch axis
            return X

    elif isinstance(space, gym.spaces.Tuple):
        def func(rng, X):
            rngs = hk.PRNGSequence(rng)
            return tuple(
                default_preprocessor(sp)(next(rngs), X[i]) for i, sp in enumerate(space.spaces))

    elif isinstance(space, gym.spaces.Dict):
        def func(rng, X):
            rngs = hk.PRNGSequence(rng)
            return {k: default_preprocessor(sp)(next(rngs), X[k]) for k, sp in space.spaces.items()}

    else:
        raise TypeError(f"unsupported space: {space.__class__.__name__}")

    return func


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

    dtype : dtype, optional

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


def diff_transform(X, dtype='float32'):
    r"""

    A helper function that implements discrete differentiation for stacked state observations. See
    :func:`diff_transform_matrix` for a detailed description.

    .. code:: python

        M = diff_transform_matrix(num_frames=X.shape[-1])
        X_transformed = np.dot(X, M)


    Parameters
    ----------
    X : ndarray

        An array whose shape is such that the last axis is the frame-stack axis, i.e.
        :code:`X.shape[-1] == num_frames`.

    Returns
    -------
    X_transformed : ndarray

        The shape is the same as the input shape, but the last axis are mixed to represent position,
        velocity, acceleration, etc.

    """
    M = diff_transform_matrix(num_frames=X.shape[-1])
    return jnp.dot(X, M)


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


def get_transition_batch(env, batch_size=1, gamma=0.9, random_seed=None):
    r"""
    Generate a single transition from the environment.

    This basically does a single step on the environment and then closes it.

    Parameters
    ----------
    env : gym environment

        A gym-style environment.

    batch_size : positive int, optional

        The desired batch size of the sample.

    random_seed : int, optional

        In order to generate the transition, we do some random sampling from the provided spaces.
        This `random_seed` set the seed for the pseudo-random number generators.

    Returns
    -------
    transition_batch : TransitionBatch

        A batch of transitions.

    """
    # import inline to avoid circular dependencies
    from ..reward_tracing import TransitionBatch
    from ._array import safe_sample

    # check types
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise TypeError(f"batch_size must be a positive int, got: {batch_size}")
    if not (isinstance(gamma, (int, float)) and 0 <= gamma <= 1):
        raise TypeError(f"gamma must be a float in the unit interval [0,1], got: {gamma}")

    rnd = onp.random.RandomState(random_seed)

    def batch_sample(space):
        max_seed = onp.iinfo('int32').max
        X = [safe_sample(space, seed=rnd.randint(max_seed)) for _ in range(batch_size)]
        return jax.tree_multimap(lambda *leaves: onp.stack(leaves, axis=0), *X)

    return TransitionBatch(
        S=batch_sample(env.observation_space),
        A=batch_sample(env.action_space),
        logP=onp.log(onp.clip(rnd.rand(batch_size), 0.01, 0.99)),
        Rn=onp.clip(rnd.randn(batch_size), -5., 5.),
        In=rnd.choice((0, gamma), batch_size),
        S_next=batch_sample(env.observation_space),
        A_next=batch_sample(env.action_space),
        logP_next=onp.log(onp.clip(rnd.rand(batch_size), 0.01, 0.99)),
        W=onp.clip(rnd.rand(batch_size) / rnd.rand(batch_size), 0.01, 100.),
    )


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

    This helper uses a slightly looser definition of scalar compared to :func:`numpy.isscalar` (and
    :func:`jax.numpy.isscalar`) in that it also considers single-item arrays to be scalars as well.

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


class StepwiseLinearFunction:
    r"""

    Stepwise linear function. The function remains flat outside of the regions defined by
    :code:`steps`.

    Parameters
    ----------
    \*steps : sequence of tuples (int, float)

        Each step :code:`(timestep, value)` fixes the output value at :code:`timestep` to the
        provided :code:`value`.

    Example
    -------
    Here's an example of the exploration schedule in a DQN agent:

    .. code::

        pi = coax.EpsilonGreedy(q, epsilon=1.0)
        epsilon = StepwiseLinearFunction((0, 1.0), (1000000, 0.1), (2000000, 0.01))

        for _ in range(num_episodes):
            pi.epsilon = epsilon(T)  # T is a global step counter
            ...

    .. image:: /_static/img/piecewise_linear_function.svg
        :alt: description
        :width: 100%
        :align: left


    Notice that the function is flat outside the interpolation range provided by :code:`steps`.


    """

    def __init__(self, *steps):
        if len(steps) < 2:
            raise TypeError("need at least two steps")
        if not all(
                isinstance(s, tuple) and len(s) == 2                          # check if pair
                and isinstance(s[0], int) and isinstance(s[1], (float, int))  # check types
                for s in steps):
            raise TypeError("all steps must be pairs (size-2 tuples) of (int, type(start_value))")
        if not all(t1 < t2 for (t1, _), (t2, _) in zip(steps, steps[1:])):  # check if consecutive
            raise ValueError(
                "steps [(t1, value), ..., (t2, value)] must be provided in ascending order, i.e. "
                "0 < t1 < t2 < ... < tn")

        self._start_value = float(steps[0][1])
        self._final_value = float(steps[-1][1])
        self._offsets = onp.array([t for t, _ in steps])
        self._intercepts = onp.array([v for _, v in steps])
        self._index = onp.arange(len(steps))
        self._slopes = onp.array([
            (v_next - v) / (t_next - t) for (t, v), (t_next, v_next) in zip(steps, steps[1:])])

    def __call__(self, timestep):
        r"""

        Return the value according to the provided schedule.

        """
        mask = self._offsets <= timestep
        if not onp.any(mask):
            return self._start_value
        if onp.all(mask):
            return self._final_value
        i = onp.max(self._index[mask])
        return self._intercepts[i] + self._slopes[i] * (timestep - self._offsets[i])


def _safe_sample(space, rnd):
    if isinstance(space, gym.spaces.Discrete):
        return rnd.randint(space.n)

    if isinstance(space, gym.spaces.MultiDiscrete):
        return onp.asarray([rnd.randint(n) for n in space.nvec])

    if isinstance(space, gym.spaces.MultiBinary):
        return rnd.randint(2, size=space.n)

    if isinstance(space, gym.spaces.Box):
        low = onp.clip(space.low, -1e9, 1e9)
        high = onp.clip(space.high, -1e9, 1e9)
        x = low + rnd.rand(*space.shape) * (high - low)
        return onp.sign(x) * onp.log(1. + onp.abs(x))  # log transform to avoid very large numbers

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


def tree_sample(pytree, rng, n=1, replace=False, axis=0, p=None):
    r"""

    Flatten and concatenate all leaves into a single flat ndarray.

    Parameters
    ----------
    pytree : a pytree with ndarray leaves

        A typical example is a pytree of model parameters (weights) or gradients with respect to
        such model params.

    rng : jax.random.PRNGKey

        A pseudo-random number generator key.

    n : int, optional

        The sample size. Note that the smaple size cannot exceed the batch size of the provided
        :code:`pytree` if :code:`with_replacement=False`.

    replace : bool, optional

        Whether to sample with replacement.

    axis : int, optional

        The axis along which to sample.

    p : 1d array, optional

        The sampling propensities.

    Returns
    -------
    arr : ndarray with ndim=1

        A single flat array.

    """
    batch_size = _check_leaf_batch_size(pytree)
    ix = jax.random.choice(rng, batch_size, shape=(n,), replace=replace, p=p)
    return jax.tree_map(lambda x: jnp.take(x, ix, axis=0), pytree)


def unvectorize(f, in_axes=0, out_axes=0):
    """

    Apply a batched function on a single instance, which effectively does the
    inverse of what :func:`jax.vmap` does.

    Parameters
    ----------
    f : callable

        A batched function.

    in_axes : int or tuple of ints, optional

        Specify the batch axes of the inputs of the function :code:`f`. If left unpsecified, this
        defaults to axis 0 for all inputs.

    out_axis: int, optional

        Specify the batch axes of the outputs of the function :code:`f`. These axes will be dropped
        by :func:`jnp.squeeze <jax.numpy.squeeze>`, i.e. dropped. If left unpsecified, this defaults
        to axis 0 for all outputs.

    Returns
    -------
    f_single : callable

        The unvectorized version of :code:`f`.

    Examples
    --------

    Haiku uses a batch-oriented design (although some components may be batch-agnostic). To create a
    function that acts on a single instance, we can use :func:`unvectorize` as follows:

    .. code:: python

        import jax.numpy as jnp
        import haiku as hk
        import coax


        def f(x_batch):
            return hk.Linear(11)(x_batch)


        rngs = hk.PRNGSequence(42)

        x_batch = jnp.zeros(shape=(3, 5))  # batch of 3 instances
        x_single = jnp.zeros(shape=(5,))   # single instance

        init, f_batch = hk.transform(f)
        params = init(next(rngs), x_batch)
        y_batch = f_batch(params, next(rngs), x_batch)
        assert y_batch.shape == (3, 11)

        f_single = coax.unvectorize(f_batch, in_axes=(None, None, 0), out_axes=0)
        y_single = f_single(params, next(rngs), x_single)
        assert y_single.shape == (11,)

    Alternatively, and perhaps more conveniently, we can unvectorize the function before doing the
    Haiku transform:

    .. code:: python

        init, f_single = hk.transform(coax.unvectorize(f))
        params = init(next(rngs), x_single)
        y_single = f_single(params, next(rngs), x_single)
        assert y_single.shape == (11,)


    """
    def f_single(*args):
        in_axes_ = in_axes
        if in_axes is None or isinstance(in_axes, int):
            in_axes_ = (in_axes,) * len(args)
        if len(args) != len(in_axes_):
            raise ValueError("number of in_axes must match the number of function inputs")
        vargs = [
            arg if axis is None else
            jax.tree_map(partial(jnp.expand_dims, axis=axis), arg)
            for arg, axis in zip(args, in_axes_)]
        out = f(*vargs)
        out_axes_ = out_axes
        if isinstance(out, tuple):
            if out_axes_ is None or isinstance(out_axes_, int):
                out_axes_ = (out_axes_,) * len(out)
            if len(out) != len(out_axes_):
                raise ValueError("number of out_axes must match the number of function outputs")
            out = tuple(
                x if axis is None else
                jax.tree_map(partial(jnp.squeeze, axis=axis), x)
                for x, axis in zip(out, out_axes_))
        elif out_axes_ is not None:
            if not isinstance(out_axes_, int):
                raise TypeError(
                    "out_axes must be an int for functions with a single output; "
                    f"got: out_axes={out_axes}")
            out = jax.tree_map(partial(jnp.squeeze, axis=out_axes), out)
        return out
    return f_single


def _check_leaf_batch_size(pytree):
    """ some boilerplate to extract the batch size with some consistency checks """
    leaf, *leaves = jax.tree_leaves(pytree)
    if not isinstance(leaf, (onp.ndarray, jnp.ndarray)) and leaf.ndim >= 1:
        raise TypeError(f"all leaves must be arrays; got type: {type(leaf)}")
    if leaf.ndim < 1:
        raise TypeError("all leaves must be at least 1d, i.e. (batch_size, ...)")
    batch_size = leaf.shape[0]
    for leaf in leaves:
        if not isinstance(leaf, (onp.ndarray, jnp.ndarray)) and leaf.ndim >= 1:
            raise TypeError(f"all leaves must be arrays; got type: {type(leaf)}")
        if leaf.ndim < 1:
            raise TypeError("all leaves must be at least 1d, i.e. (batch_size, ...)")
        if leaf.shape[0] != batch_size:
            raise TypeError("all leaves must have the same batch_size")
    return batch_size


def stack_trees(*trees):
    """
    Stack
    Parameters
    ----------
    trees : sequence of pytrees with ndarray leaves
        A typical example are pytrees containing the parameters and function states of
        a model that should be used in a function which is vectorized by `jax.vmap`. The trees
        have to have the same pytree structure.
    Returns
    -------
    pytree : pytree with ndarray leaves
        A tuple of pytrees.
    """
    return jax.tree_util.tree_multimap(lambda *args: jnp.stack(args), *zip(*trees))
