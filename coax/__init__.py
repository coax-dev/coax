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

__version__ = '0.1.6'


# expose specific classes and functions
from ._core.v import V
from ._core.q import Q
from ._core.policy import Policy
from ._core.worker import Worker
from ._core.reward_function import RewardFunction
from ._core.transition_model import TransitionModel
from ._core.stochastic_v import StochasticV
from ._core.stochastic_q import StochasticQ
from ._core.stochastic_transition_model import StochasticTransitionModel
from ._core.stochastic_reward_function import StochasticRewardFunction
from ._core.value_based_policy import EpsilonGreedy, BoltzmannPolicy
from ._core.random_policy import RandomPolicy
from ._core.successor_state_q import SuccessorStateQ
from .utils import safe_sample, render_episode

# pre-load submodules
from . import experience_replay
from . import model_updaters
from . import policy_objectives
from . import proba_dists
from . import regularizers
from . import reward_tracing
from . import td_learning
from . import typing
from . import utils
from . import value_losses
from . import wrappers


__all__ = (

    # classes and functions
    'V',
    'Q',
    'Policy',
    'Worker',
    'RewardFunction',
    'TransitionModel',
    'StochasticV',
    'StochasticQ',
    'StochasticRewardFunction',
    'StochasticTransitionModel',
    'EpsilonGreedy',
    'BoltzmannPolicy',
    'RandomPolicy',
    'SuccessorStateQ',
    'safe_sample',
    'render_episode',

    # modules
    'experience_replay',
    'model_updaters',
    'policy_objectives',
    'proba_dists',
    'regularizers',
    'reward_tracing',
    'td_learning',
    'typing',
    'utils',
    'value_losses',
    'wrappers',
)


# -----------------------------------------------------------------------------
# register envs
# -----------------------------------------------------------------------------

import gym

if 'ConnectFour-v0' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['ConnectFour-v0']

gym.envs.register(
    id='ConnectFour-v0',
    entry_point='coax.envs:ConnectFourEnv',
)


if 'FrozenLakeNonSlippery-v0' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['FrozenLakeNonSlippery-v0']

gym.envs.register(
    id='FrozenLakeNonSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=20,
    reward_threshold=0.99,
)

del gym
