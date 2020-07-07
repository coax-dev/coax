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
