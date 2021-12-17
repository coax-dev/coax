from typing import TypeVar, Union, Sequence, Callable, Tuple


__all__ = (
    'Batch',
    'SpaceElement',
    'Observation',
    'Action'
)


Batch = Sequence                            # annotate batched quantities
Observation = TypeVar('Observation')        # a state observation
Action = TypeVar('Action')                  # an action
SpaceElement = Union[Observation, Action]   # element of a gym-style space
LogPropensity = TypeVar('LogPropensity')    # an action


Policy = Callable[
    [Observation, bool],
    Union[Action, Tuple[Action, LogPropensity]]
]
