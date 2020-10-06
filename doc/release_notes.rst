Release Notes
=============


v0.1.3
------

Implemented Distributional RL algorithm:

- Added two new methods to all proba_dists: :attr:`mean` and :attr:`affine_transform`, see
  :mod:`coax.proba_dists`.
- Made TD-learning updaters compatible with :class:`coax.StochasticV` and :class:`coax.StochasticQ`.
- Made value-based policies compatible with :class:`coax.StochasticQ`.


v0.1.2
------

First version to go public.
