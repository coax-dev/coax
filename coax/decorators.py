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

Decorators
==========

This module provides some decorators that can be used to reduce some verbosity in the definition of
function approximators.


.. autosummary::
    :nosignatures:

    coax.decorators.policy
    coax.decorators.value_q
    coax.decorators.value_v

"""

from ._core.value_v import V
from ._core.value_q import Q
from ._core.policy import Policy

__all__ = (
    'policy',
    'value_q',
    'value_v',
)


def value_v(env, random_seed=None):
    r"""

    This decorator is implements some syntactic sugar:

    .. code:: python

        @coax.value_v(env)
        def v(S, is_training):
            value = hk.Sequential((
                ...
            ))
            return value(S)

    which is a short-hand implementation of:

    .. code:: python

        def func(S, is_training):
            value = hk.Sequential((
                ...
            ))
            return value(S)

        v = coax.V(func, env.observation_space)

    """
    def decorator(func):
        return V(func, env.observation_space, random_seed)
    return decorator


def value_q(env, random_seed=None):
    r"""

    This decorator is implements some syntactic sugar:

    .. code:: python

        @coax.value_q(env)
        def q(S, A, is_training):
            value = hk.Sequential((
                ...
            ))
            return value(S)

    which is a short-hand implementation of:

    .. code:: python

        def func(S, A, is_training):
            value = hk.Sequential((
                ...
            ))
            return value(S)

        q = coax.Q(func, env.observation_space, env.action_space)

    """
    def decorator(func):
        return Q(func, env.observation_space, env.action_space, random_seed)
    return decorator


def policy(env, random_seed=None):
    r"""

    This decorator is implements some syntactic sugar:

    .. code:: python

        @coax.policy(env)
        def pi(S, A, is_training):
            value = hk.Sequential((
                ...
            ))
            return value(S)

    which is a short-hand implementation of:

    .. code:: python

        def func(S, is_training):
            logits = hk.Sequential((
                ...
            ))
            return {'logits': logits(S)}  # note: this example is specific to discrete actions

        pi = coax.Policy(func, env.observation_space, env.action_space)

    """
    def decorator(func):
        return Policy(func, env.observation_space, env.action_space, random_seed)
    return decorator
