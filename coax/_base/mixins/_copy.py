from copy import deepcopy, copy


class CopyMixin:
    def copy(self, deep=False):
        r"""

        Create a copy of the current instance.

        Parameters
        ----------
        deep : bool, optional

            Whether the copy should be a deep copy.

        Returns
        -------
        copy

            A deep copy of the current instance.

        """
        return deepcopy(self) if deep else copy(self)
