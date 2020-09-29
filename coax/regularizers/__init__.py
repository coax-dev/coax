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

r"""
.. autosummary::
    :nosignatures:

    coax.regularizers.EntropyRegularizer
    coax.regularizers.KLDivRegularizer

----

Regularizers
============

This is a collection of regularizers that can be used to put soft constraints on stochastic function
approximators. These is typically added to the loss/objective to avoid premature exploitation of a
policy.


Object Reference
----------------

..autoclass:: coax.regularizers.EntropyRegularizer
..autoclass:: coax.regularizers.KLDivRegularizer

"""

from ._entropy import Regularizer
from ._entropy import EntropyRegularizer
from ._kl_div import KLDivRegularizer


__all__ = (
    'Regularizer',
    'EntropyRegularizer',
    'KLDivRegularizer',
)
