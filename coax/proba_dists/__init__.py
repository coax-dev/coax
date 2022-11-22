r"""
.. autosummary::
    :nosignatures:

    coax.proba_dists.ProbaDist
    coax.proba_dists.CategoricalDist
    coax.proba_dists.NormalDist
    coax.proba_dists.DiscretizedIntervalDist
    coax.proba_dists.EmpiricalQuantileDist
    coax.proba_dists.SquashedNormalDist

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
.. autoclass:: coax.proba_dists.EmpiricalQuantileDist
.. autoclass:: coax.proba_dists.SquashedNormalDist


"""

from ._composite import ProbaDist
from ._categorical import CategoricalDist
from ._normal import NormalDist
from ._discretized_interval import DiscretizedIntervalDist
from ._empirical_quantile import EmpiricalQuantileDist
from ._squashed_normal import SquashedNormalDist


__all__ = (
    'ProbaDist',
    'CategoricalDist',
    'NormalDist',
    'DiscretizedIntervalDist',
    'EmpiricalQuantileDist',
    'SquashedNormalDist',
)
