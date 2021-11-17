
class ValueTransform:
    r"""

    Abstract base class for value transforms. See
    :class:`coax.value_transforms.LogTransform` for a specific implementation.

    """
    __slots__ = ('_transform_func', '_inverse_func')

    def __init__(self, transform_func, inverse_func):
        self._transform_func = transform_func
        self._inverse_func = inverse_func

    @property
    def transform_func(self):
        r"""

        The transformation function :math:`x\mapsto y=f(x)`.

        Parameters
        ----------
        x : ndarray

            The values in their original representation.

        Returns
        -------
        y : ndarray

            The values in their transformed representation.

        """
        return self._transform_func

    @property
    def inverse_func(self):
        r"""

        The inverse transformation function :math:`y\mapsto x=f^{-1}(y)`.

        Parameters
        ----------
        y : ndarray

            The values in their transformed representation.

        Returns
        -------
        x : ndarray

            The values in their original representation.

        """
        return self._inverse_func

    def __iter__(self):
        return iter((self.transform_func, self.inverse_func))
