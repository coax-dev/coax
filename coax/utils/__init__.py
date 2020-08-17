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

Utilities
=========

This is a collection of utility (helper) functions used throughout the package.


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.utils.argmax
    coax.utils.argmin
    coax.utils.batch_to_single
    coax.utils.check_array
    coax.utils.clipped_logit
    coax.utils.diff_transform_matrix
    coax.utils.docstring
    coax.utils.double_relu
    coax.utils.enable_logging
    coax.utils.generate_gif
    coax.utils.get_env_attr
    coax.utils.get_grads_diagnostics
    coax.utils.get_magnitude_quantiles
    coax.utils.get_transition
    coax.utils.has_env_attr
    coax.utils.idx
    coax.utils.is_policy
    coax.utils.is_qfunction
    coax.utils.is_vfunction
    coax.utils.isscalar
    coax.utils.merge_dicts
    coax.utils.OrnsteinUhlenbeckNoise
    coax.utils.reload_recursive
    coax.utils.render_episode
    coax.utils.safe_sample
    coax.utils.single_to_batch
    coax.utils.strip_env_recursive
    coax.utils.StrippedEnv
    coax.utils.tree_ravel


"""

from ._action_noise import (
    OrnsteinUhlenbeckNoise,
)
from ._array import (
    argmax,
    argmin,
    batch_to_single,
    check_array,
    clipped_logit,
    diff_transform_matrix,
    double_relu,
    get_magnitude_quantiles,
    idx,
    isscalar,
    merge_dicts,
    single_to_batch,
    safe_sample,
    get_grads_diagnostics,
    tree_ravel,
)
from ._misc import (
    docstring,
    enable_logging,
    generate_gif,
    get_env_attr,
    get_transition,
    has_env_attr,
    is_policy,
    is_qfunction,
    is_vfunction,
    render_episode,
    reload_recursive,
    StrippedEnv,
    strip_env_recursive,
)


__all__ = (
    'argmax',
    'argmin',
    'batch_to_single',
    'check_array',
    'clipped_logit',
    'diff_transform_matrix',
    'docstring',
    'double_relu',
    'enable_logging',
    'generate_gif',
    'get_env_attr',
    'get_transition',
    'get_grads_diagnostics',
    'get_magnitude_quantiles',
    'has_env_attr',
    'idx',
    'is_policy',
    'is_qfunction',
    'is_vfunction',
    'isscalar',
    'merge_dicts',
    'OrnsteinUhlenbeckNoise',
    'reload_recursive',
    'render_episode',
    'safe_sample',
    'single_to_batch',
    'StrippedEnv',
    'strip_env_recursive',
    'tree_ravel',
)
