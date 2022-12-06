r"""

Utilities
=========

This is a collection of utility (helper) functions used throughout the package.

.. autosummary::
    :nosignatures:

    coax.utils.OrnsteinUhlenbeckNoise
    coax.utils.StepwiseLinearFunction
    coax.utils.SegmentTree
    coax.utils.SumTree
    coax.utils.MinTree
    coax.utils.MaxTree
    coax.utils.argmax
    coax.utils.argmin
    coax.utils.batch_to_single
    coax.utils.check_array
    coax.utils.check_preprocessors
    coax.utils.clipped_logit
    coax.utils.default_preprocessor
    coax.utils.diff_transform
    coax.utils.diff_transform_matrix
    coax.utils.docstring
    coax.utils.double_relu
    coax.utils.dump
    coax.utils.dumps
    coax.utils.enable_logging
    coax.utils.generate_gif
    coax.utils.get_env_attr
    coax.utils.get_grads_diagnostics
    coax.utils.get_magnitude_quantiles
    coax.utils.get_transition_batch
    coax.utils.has_env_attr
    coax.utils.idx
    coax.utils.is_policy
    coax.utils.is_qfunction
    coax.utils.is_reward_function
    coax.utils.is_stochastic
    coax.utils.is_transition_model
    coax.utils.is_vfunction
    coax.utils.isscalar
    coax.utils.jit
    coax.utils.load
    coax.utils.loads
    coax.utils.make_dmc
    coax.utils.merge_dicts
    coax.utils.pretty_print
    coax.utils.pretty_repr
    coax.utils.quantiles
    coax.utils.quantiles_uniform
    coax.utils.quantile_cos_embedding
    coax.utils.reload_recursive
    coax.utils.render_episode
    coax.utils.safe_sample
    coax.utils.single_to_batch
    coax.utils.stack_trees
    coax.utils.sync_shared_params
    coax.utils.tree_ravel
    coax.utils.unvectorize


Object Reference
----------------

.. autoclass:: coax.utils.OrnsteinUhlenbeckNoise
.. autoclass:: coax.utils.StepwiseLinearFunction
.. autoclass:: coax.utils.SegmentTree
.. autoclass:: coax.utils.SumTree
.. autoclass:: coax.utils.MinTree
.. autoclass:: coax.utils.MaxTree
.. autofunction:: coax.utils.argmax
.. autofunction:: coax.utils.argmin
.. autofunction:: coax.utils.batch_to_single
.. autofunction:: coax.utils.check_array
.. autofunction:: coax.utils.check_preprocessors
.. autofunction:: coax.utils.clipped_logit
.. autofunction:: coax.utils.default_preprocessor
.. autofunction:: coax.utils.diff_transform
.. autofunction:: coax.utils.diff_transform_matrix
.. autofunction:: coax.utils.docstring
.. autofunction:: coax.utils.double_relu
.. autofunction:: coax.utils.dump
.. autofunction:: coax.utils.dumps
.. autofunction:: coax.utils.enable_logging
.. autofunction:: coax.utils.generate_gif
.. autofunction:: coax.utils.get_env_attr
.. autofunction:: coax.utils.get_grads_diagnostics
.. autofunction:: coax.utils.get_magnitude_quantiles
.. autofunction:: coax.utils.get_transition_batch
.. autofunction:: coax.utils.has_env_attr
.. autofunction:: coax.utils.idx
.. autofunction:: coax.utils.is_policy
.. autofunction:: coax.utils.is_qfunction
.. autofunction:: coax.utils.is_reward_function
.. autofunction:: coax.utils.is_stochastic
.. autofunction:: coax.utils.is_transition_model
.. autofunction:: coax.utils.is_vfunction
.. autofunction:: coax.utils.isscalar
.. autofunction:: coax.utils.jit
.. autofunction:: coax.utils.load
.. autofunction:: coax.utils.loads
.. autofunction:: coax.utils.make_dmc
.. autofunction:: coax.utils.merge_dicts
.. autofunction:: coax.utils.pretty_print
.. autofunction:: coax.utils.pretty_repr
.. autofunction:: coax.utils.quantiles
.. autofunction:: coax.utils.quantiles_uniform
.. autofunction:: coax.utils.quantile_cos_embedding
.. autofunction:: coax.utils.reload_recursive
.. autofunction:: coax.utils.render_episode
.. autofunction:: coax.utils.safe_sample
.. autofunction:: coax.utils.single_to_batch
.. autofunction:: coax.utils.stack_trees
.. autofunction:: coax.utils.sync_shared_params
.. autofunction:: coax.utils.tree_ravel
.. autofunction:: coax.utils.unvectorize

"""

from ._action_noise import OrnsteinUhlenbeckNoise
from ._array import (
    StepwiseLinearFunction,
    argmax,
    argmin,
    batch_to_single,
    check_array,
    check_preprocessors,
    chunks_pow2,
    clipped_logit,
    default_preprocessor,
    diff_transform,
    diff_transform_matrix,
    double_relu,
    get_grads_diagnostics,
    get_magnitude_quantiles,
    get_transition_batch,
    idx,
    isscalar,
    merge_dicts,
    safe_sample,
    single_to_batch,
    stack_trees,
    sync_shared_params,
    tree_ravel,
    unvectorize,
)
from ._jit import jit
from ._misc import (
    docstring,
    dump,
    dumps,
    enable_logging,
    generate_gif,
    get_env_attr,
    has_env_attr,
    is_policy,
    is_qfunction,
    is_reward_function,
    is_stochastic,
    is_transition_model,
    is_vfunction,
    load,
    loads,
    pretty_print,
    pretty_repr,
    reload_recursive,
    render_episode,
)
from ._segment_tree import SegmentTree, SumTree, MinTree, MaxTree
from ._quantile_funcs import quantiles, quantiles_uniform, quantile_cos_embedding
from ._dmc_gym import make_dmc


__all__ = (
    'StepwiseLinearFunction',
    'OrnsteinUhlenbeckNoise',
    'SegmentTree',
    'SumTree',
    'MinTree',
    'MaxTree',
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
    'docstring',
    'double_relu',
    'dump',
    'dumps',
    'enable_logging',
    'generate_gif',
    'get_env_attr',
    'get_grads_diagnostics',
    'get_magnitude_quantiles',
    'get_transition_batch',
    'has_env_attr',
    'idx',
    'is_policy',
    'is_qfunction',
    'is_reward_function',
    'is_stochastic',
    'is_transition_model',
    'is_vfunction',
    'isscalar',
    'jit',
    'load',
    'loads',
    'make_dmc',
    'merge_dicts',
    'pretty_print',
    'pretty_repr',
    'quantiles',
    'quantiles_uniform',
    'quantile_cos_embedding',
    'reload_recursive',
    'render_episode',
    'safe_sample',
    'single_to_batch',
    'stack_trees',
    'sync_shared_params',
    'tree_ravel',
    'unvectorize',
)
