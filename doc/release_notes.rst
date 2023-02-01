Release Notes
=============


If you need any of the features from the pre-release version listed under "Upcoming" you can just install coax from the **main** branch:

.. code::
    
    $ pip install git+https://github.com/coax-dev/coax.git@main


Upcoming
--------

* ...


v0.1.13
-------

* Switch from legacy ``gym`` to ``gymnasium`` (`#40 <https://github.com/coax-dev/coax/issues/40>`_)
* Upgrade dependencies.
 

v0.1.12
-------

* Add DeepMind Control Suite example (`#29 <https://github.com/coax-dev/coax/pull/29>`_); see :doc:`/examples/dmc/sac`.
* Add :func:`coax.utils.sync_shared_params` utility; example in :doc:`A2C stub </examples/stubs/a2c>`.
* Improved performance for replay buffer (`#25 <https://github.com/coax-dev/coax/pull/25>`_)
* Bug fix: random_seed in _prioritized (`#24 <https://github.com/coax-dev/coax/pull/24>`_)
* Update to new Jax API (`#27 <https://github.com/coax-dev/coax/pull/27>`_)
* Add Update to ``gym==0.26.x`` (`#28 <https://github.com/coax-dev/coax/pull/28>`_).
* Bug fix: set logging level on ``TrainMonitor.logger`` itself (`550a965 <https://github.com/coax-dev/coax/commit/550a965d17002bf552ab2fbea49801c65b322c7b>_`).
* Bug fix: fix affine transform for composite distributions (`48ca9ce <https://github.com/coax-dev/coax/commit/48ca9ced42123e906969076dff88540b98e6d0bb>_`)
* Bug fix: `#33 <https://github.com/coax-dev/coax/issues/33>`_


v0.1.11
-------

* Bug fix: `#21 <https://github.com/coax-dev/coax/issues/21>`_
* Fix deprecation warnings from using ``jax.tree_multimap`` and ``gym.envs.registry.env_specs``.
* Upgrade dependencies.


v0.1.10
-------

* Bug fixes: `#16 <https://github.com/coax-dev/coax/issues/16>`_
* Replace old ``jax.ops.index*`` scatter operations with the new :attr:`jax.numpy.ndarray.at` interface.
* Upgrade dependencies.


v0.1.9
------

Bumped version to drop hard dependence on ``ray``.


v0.1.8
------

Implemented stochastic q-learning using quantile regression in :class:`coax.StochasticQ`, see example: :doc:`IQN <examples/cartpole/iqn>`

* Use :func:`coax.utils.quantiles` for equally spaced quantile fractions as in QR-DQN.
* Use :func:`coax.utils.quantiles_uniform` for uniformly sampled quantile fractions as in IQN.


v0.1.7
------

This is not much of a release. It's only really the dependencies that were updated.


v0.1.6
------

* Add basic support for distributed agents, see example: :doc:`Ape-X DQN <examples/atari/apex_dqn>`
* Fixed issues with serialization of jit-compiled functions, see `jax#5043 <https://github.com/google/jax/issues/5043>`_ and `jax#5153 <https://github.com/google/jax/pull/5153#issuecomment-755930540>`_
* Add support for sample weights in reward tracers


v0.1.5
------

* Implemented :class:`coax.td_learning.SoftQLearning`.
* Added soft q-learning :doc:`stub <examples/stubs/soft_qlearning>` and
  :doc:`script <examples/atari/dqn_soft>`.
* Added serialization utils: :func:`coax.utils.dump`, :func:`coax.utils.dumps`, :func:`coax.utils.load`, :func:`coax.utils.loads`.


v0.1.4
------

Implemented Prioritized Experience Replay:

* Implemented :class:`SegmentTree <coax.experience_replay.SegmentTree>` that allows for *batched*
  updating.
* Implemented :class:`SumTree <coax.experience_replay.SumTree>` subclass that allows for *batched*
  weighted sampling.
* Drop TransitionSingle (only use :class:`TransitionBatch <coax.reward_tracing.TransitionBatch>`
  from now on).
* Added :func:`TransitionBatch.from_single <coax.reward_tracing.TransitionBatch.from_single>`
  constructor.
* Added :attr:`TransitionBatch.idx <coax.reward_tracing.TransitionBatch.idx>` field to identify
  specific transitions.
* Added :attr:`TransitionBatch.W <coax.reward_tracing.TransitionBatch.W>` field to collect sample
  weights
* Made all :mod:`td_learning <coax.td_learning>` and :mod:`policy_objectives
  <coax.policy_objectives>` updaters compatible with :attr:`TransitionBatch.W
  <coax.reward_tracing.TransitionBatch.W>`
* Implemented the :class:`PrioritizedReplayBuffer <coax.experience_replay.PrioritizedReplayBuffer>`
  class itself.
* Added scripts and notebooks: :doc:`agent stub <examples/stubs/dqn_per>` and :doc:`pong
  <examples/atari/dqn_per>`.


Other utilities:

* Added :class:`FrameStacking <coax.wrappers.FrameStacking>` wrapper that respects the
  :mod:`gym.space` API and is compatible with the :mod:`jax.tree_util` module.
* Added data summary (min, median, max) for arrays in :class:`pretty_repr <coax.utils.pretty_repr>`
  util.
* Added :class:`StepwiseLinearFunction <coax.utils.StepwiseLinearFunction>` utility, which is handy
  for hyperparameter schedules, see example usage :doc:`here <examples/stubs/dqn_per>`.


v0.1.3
------

Implemented Distributional RL algorithm:

* Added two new methods to all proba_dists: :attr:`mean` and :attr:`affine_transform`, see
  :mod:`coax.proba_dists`.
* Made TD-learning updaters compatible with :class:`coax.StochasticV` and :class:`coax.StochasticQ`.
* Made value-based policies compatible with :class:`coax.StochasticQ`.


v0.1.2
------

First version to go public.
