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

import numpy as onp


__all__ = (
    'SegmentTree',
    'SumTree',
    'MinTree',
    'MaxTree',
)


class SegmentTree:
    r"""

    A `segment tree <https://en.wikipedia.org/wiki/Segment_tree>`_ data structure that allows
    for batched updating and batched partial-range (segment) reductions.

    Parameters
    ----------
    capacity : positive int

        Number of values to accommodate.

    reducer : function

        The reducer function: :code:`(float, float) -> float`.

    init_value : float

        The unit element relative to the reducer function. Some typical examples are: 0 if reducer
        is :data:`add <numpy.add>`, 1 for :data:`multiply <numpy.multiply>`, :math:`-\infty` for
        :data:`maximum <numpy.maximum>`, :math:`\infty` for :data:`minimum <numpy.minimum>`.

    """
    def __init__(self, capacity, reducer, init_value):
        self.capacity = capacity
        self.reducer = reducer
        self.init_value = float(init_value)
        self._height = int(onp.ceil(onp.log2(capacity))) + 1  # the +1 is for the values themselves
        self._arr = onp.full(shape=(2 ** self.height - 1), fill_value=self.init_value)

    @property
    def height(self):
        r""" The height of the tree :math:`h\sim\log(\text{capacity})`. """
        return self._height

    @property
    def root_value(self):
        r"""

        The aggregated value, equivalent to
        :func:`reduce(reducer, values, init_value) <functools.reduce>`.

        """
        return self._arr[0]

    @property
    def values(self):
        r""" The values stored at the leaves of the tree. """
        return self[...]

    @values.setter
    def values(self, new_values):
        self[...] = new_values

    def __array__(self):
        return self.values

    def __getitem__(self, idx):
        idx = self._check_level_idx(self.height - 1, idx, self.capacity)
        return self._arr[idx]

    def __setitem__(self, idx, values):
        idx = self._check_level_idx(self.height - 1, idx, self.capacity)
        self._arr[idx] = values                                # update leaf-node values
        level_offset = 2 ** (self.height - 1) - 1               # number of nodes in higher levels
        for level in range(self.height - 2, -1, -1):
            level_idx = onp.unique((idx - level_offset) // 2)  # level index (without offset)
            left_child_idx = level_idx * 2 + level_offset      # indices for left children
            right_child_idx = left_child_idx + 1               # indices for right children
            level_offset = 2 ** level - 1
            idx = level_idx + level_offset                     # parent indices
            self._arr[idx] = self.reducer(self._arr[left_child_idx], self._arr[right_child_idx])

    def partial_reduce(self, start=0, stop=None):
        r"""

        Reduce values over a partial range of indices. This is an efficient, batched implementation
        of :func:`reduce(reducer, values[state:stop], init_value) <functools.reduce>`.

        Parameters
        ----------
        start : int or array of ints

            The lower bound of the range (inclusive).

        stop : int or array of ints, optional

            The lower bound of the range (exclusive). If left unspecified, this defaults to
            :attr:`height`.

        Returns
        -------
        value : float

            The result of the partial reduction.

        """
        # NOTE: This is an iterative implementation, which is a lot uglier than a recursive one.
        # The reason why we use an iterative approach is that it's easier for batch-processing.

        # i and j are 1d arrays (indices for self._arr)
        i, j = self._check_start_stop_to_i_j(start, stop)

        # trivial case
        done = (i == j)
        if done.all():
            return self._arr[i]

        # left/right accumulators (mask one of them to avoid over-counting if i == j)
        a, b = self._arr[i], onp.where(done, self.init_value, self._arr[j])

        # number of nodes in higher levels
        level_offset = 2 ** (self.height - 1) - 1

        # we start from the leaves and work up towards the root
        for level in range(self.height - 2, -1, -1):

            # get parent indices
            level_offset_parent = 2 ** level - 1
            i_parent = (i - level_offset) // 2 + level_offset_parent
            j_parent = (j - level_offset) // 2 + level_offset_parent

            # stop when we have a shared parent (possibly the root node, but not necessarily)
            done |= (i_parent == j_parent)
            if done.all():
                return self.reducer(a, b)

            # only accumulate right-child value if 'i' was a left child of 'i_parent'
            a = onp.where((i % 2 == 1) & ~done, self.reducer(a, self._arr[i + 1]), a)

            # only accumulate left-child value if 'j' was a right child of 'j_parent'
            b = onp.where((j % 2 == 0) & ~done, self.reducer(b, self._arr[j - 1]), b)

            # prepare for next loop
            i, j, level_offset = i_parent, j_parent, level_offset_parent

        assert False, 'this point should not be reached'

    def __repr__(self):
        s = ""
        for level in range(self.height):
            idx = self._check_level_idx(level, ...)
            s += f"\n  level={level} : {repr(self._arr[idx])}"
        return f"{type(self).__name__}({s})"

    def _check_level(self, level):
        if level < -self.height or level >= self.height:
            raise IndexError(f"tree level index {level} out of range; tree height: {self.height}")
        return level % self.height

    def _check_level_idx(self, level, idx, level_size=None):
        """ some boilerplate to transform an index relative to a tree level into a global index """
        i = 2 ** level - 1                                    # level offset
        n = 2 ** level if level_size is None else level_size  # level size

        if idx is None or idx is Ellipsis:
            return onp.arange(i, i + n)

        if isinstance(idx, slice):
            if not (idx.step is None or idx.step == 1):
                raise NotImplementedError("reverse indexing not implemented; feature request?")
            if not (idx.start is None or 0 <= idx.start < n):
                raise IndexError(f"index slice.start out of range: {idx.start} not in [0, {n})")
            if not (idx.stop is None or 0 < idx.stop <= n):
                raise IndexError(f"index slice.stop out of range: {idx.stop} not in (0, {n}]")
            start = i if idx.start is None else idx.start + i
            stop = i + n if idx.stop is None else idx.stop + i
            return onp.arange(start, stop)

        idx = onp.asarray(idx, dtype='int32')
        if onp.any(idx >= n) or onp.any(idx < -level_size):
            raise IndexError("one of more entries in idx are out or range")
        return idx % level_size + i  # offset

    def _check_start_stop_to_i_j(self, start, stop):
        """ some boiler plate to turn (start, stop) into left/right index arrays (i, j) """
        start_orig, stop_orig = start, stop
        if isinstance(start, int):
            start = onp.array([start])
        if not (isinstance(start, onp.ndarray)
                and start.ndim == 1
                and onp.issubdtype(start.dtype, onp.integer)):
            raise TypeError("'start' must be an int or a 1d array of ints array")

        if stop is None:
            stop = onp.full_like(start, self.capacity)
        if isinstance(stop, int):
            stop = onp.full_like(start, stop)
        if not (isinstance(stop, onp.ndarray)
                and stop.ndim == 1
                and onp.issubdtype(stop.dtype, onp.integer)):
            raise TypeError("'stop' must be an int or a 1d array of ints array")

        if start.size == 1 and stop.size > 1:
            start = onp.full_like(stop, start[0])

        if start.shape != stop.shape:
            raise ValueError(
                f"shapes must be equal, got: start.shape: {start.shape}, stop.shape: {stop.shape}")

        i = self._check_level_idx(self.height - 1, start, self.capacity)
        j = self._check_level_idx(self.height - 1, stop - 1, self.capacity)

        if not onp.all(i <= j):
            raise IndexError(
                f"inconsistent ranges detected from (start, stop) = ({start_orig}, {stop_orig})")

        return i, j


class SumTree(SegmentTree):
    r"""

    A sum-tree data structure that allows for batched updating and batched weighted sampling.

    Both update and sampling operations have a time complexity of :math:`\mathcal{O}(\log N)` and a
    memory footprint of :math:`\mathcal{O}(N)`, where :math:`N` is the length of the underlying
    :attr:`values`.

    Parameters
    ----------
    capacity : positive int

        Number of values to accommodate.

    reducer : function

        The reducer function: :code:`(float, float) -> float`.

    init_value : float

        The unit element relative to the reducer function. Some typical examples are: 0 if
        reducer is :func:`operator.add`, 1 for :func:`operator.mul`, :math:`-\infty` for
        :func:`max`, :math:`\infty` for :func:`min`.

    """
    def __init__(self, capacity, random_seed=None):
        super().__init__(capacity=capacity, reducer=onp.add, init_value=0)
        self.random_seed = random_seed

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, new_random_seed):
        self._rnd = onp.random.RandomState(new_random_seed)
        self._random_seed = new_random_seed

    def sample(self, n):
        r"""

        Sample array indices using weighted sampling, where the sample weights are proprotional to
        the values stored in :attr:`values`.

        Parameters
        ----------
        n : positive int

            The number of samples to return.

        Returns
        -------
        idx : array of ints

            The sampled indices, shape: (n,)

        """
        if not (isinstance(n, int) and n > 0):
            raise TypeError("n must be a positive integer")

        return self.inverse_cdf(self._rnd.rand(n))

    def inverse_cdf(self, u):
        r"""

        Inverse of the cumulative distribution function (CDF) of the categorical distribution
        :math:`\text{Cat}(p)`, where :math:`p` are the normalized values :math:`p_i=`
        :attr:`values[i] / sum(values) <values>`.

        This function provides the machinery for the :attr:`sample` method.

        Parameters
        ----------
        u : float or 1d array of floats

            One of more numbers :math:`u\in[0,1]`. These are typically sampled from
            :math:`\text{Unif([0, 1])}`.

        Returns
        -------
        idx : array of ints

            The indices associated with :math:`u`, shape: (n,)

        """
        # NOTE: This is an iterative implementation, which is a lot uglier than a recursive one.
        # The reason why we use an iterative approach is that it's easier for batch-processing.

        # init (will be updated in loop)
        u, isscalar = self._check_u(u)
        values = u * self.root_value
        idx = onp.zeros_like(values, dtype='int32')  # this is ultimately what we'll returned
        level_offset_parent = 0                      # number of nodes in levels above parent

        # iterate down, from the root to the leaves
        for level in range(1, self.height):

            # get child indices
            level_offset = 2 ** level - 1
            left_child_idx = (idx - level_offset_parent) * 2 + level_offset
            right_child_idx = left_child_idx + 1

            # update (idx, values, level_offset_parent)
            left_child_values = self._arr[left_child_idx]
            pick_left_child = left_child_values > values
            idx = onp.where(pick_left_child, left_child_idx, right_child_idx)
            values = onp.where(pick_left_child, values, values - left_child_values)
            level_offset_parent = level_offset

        idx = idx - level_offset_parent
        return idx[0] if isscalar else idx

    def _check_u(self, u):
        """ some boilerplate to check validity of 'u' array """
        isscalar = False
        if isinstance(u, (float, int)):
            u = onp.array([u], dtype='float')
            isscalar = True
        if isinstance(u, list) and all(isinstance(x, (float, int)) for x in u):
            u = onp.asarray(u, dtype='float')
        if not (isinstance(u, onp.ndarray)
                and u.ndim == 1 and onp.issubdtype(u.dtype, onp.floating)):
            raise TypeError("'u' must be a float or a 1d array of floats")
        if onp.any(u > 1) or onp.any(u < 0):
            raise ValueError("all values in 'u' must lie in the unit interval [0, 1]")
        return u, isscalar


class MinTree(SegmentTree):
    r"""

    A min-tree data structure, which is a :class:`SegmentTree` whose reducer is :data:`minimum
    <numpy.minimum>`.

    Parameters
    ----------
    capacity : positive int

        Number of values to accommodate.

    """
    def __init__(self, capacity):
        super().__init__(capacity=capacity, reducer=onp.minimum, init_value=float('inf'))


class MaxTree(SegmentTree):
    r"""

    A max-tree data structure, which is a :class:`SegmentTree` whose reducer is :data:`maximum
    <numpy.maximum>`.

    Parameters
    ----------
    capacity : positive int

        Number of values to accommodate.

    """
    def __init__(self, capacity):
        super().__init__(capacity=capacity, reducer=onp.maximum, init_value=-float('inf'))
