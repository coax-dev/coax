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

    coax.proba_dists.ProbaDist
    coax.proba_dists.CategoricalDist
    coax.proba_dists.NormalDist
    coax.proba_dists.DiscretizedIntervalDist

-----

Probability Distributions
=========================

This is a collection of **differentiable** probability distributions used throughout the package.


Object Reference
----------------

.. autoclass:: coax.proba_dists.ProbaDist
.. autoclass:: coax.proba_dists.CategoricalDist
.. autoclass:: coax.proba_dists.NormalDist
.. autoclass:: coax.proba_dists.DiscretizedIntervalDist


"""

from ._composite import ProbaDist
from ._categorical import CategoricalDist
from ._normal import NormalDist
from ._discretized_interval import DiscretizedIntervalDist


__all__ = (
    'ProbaDist',
    'CategoricalDist',
    'NormalDist',
    'DiscretizedIntervalDist',
)
