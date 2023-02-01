__version__ = '0.1.13'

# fall back to legacy gym if gymnasium is unavailable
try:
    import gymnasium as _gymnasium
except ImportError:
    import sys
    import warnings
    warnings.warn("Cannot import 'gymnasium'; attempting to fall back to legacy 'gym'.")
    import gym as _gymnasium  # Don't catch ImportError here; we need gymnasium or gym.
    sys.modules['gymnasium'] = _gymnasium
    del sys, warnings  # Keep namespace clean.

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
from .utils import safe_sample, render_episode, unvectorize

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
    'unvectorize',

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

if 'ConnectFour-v0' in _gymnasium.envs.registry:
    del _gymnasium.envs.registry['ConnectFour-v0']

_gymnasium.envs.register(
    id='ConnectFour-v0',
    entry_point='coax.envs:ConnectFourEnv',
)


if 'FrozenLakeNonSlippery-v0' in _gymnasium.envs.registry:
    del _gymnasium.envs.registry['FrozenLakeNonSlippery-v0']

_gymnasium.envs.register(
    id='FrozenLakeNonSlippery-v0',
    entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=20,
    reward_threshold=0.99,
)

del _gymnasium  # Keep namespace clean.
