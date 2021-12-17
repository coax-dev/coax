from abc import ABC, abstractmethod
from typing import Any, Tuple, NamedTuple

import jax
import haiku as hk
from gym.spaces import Space

from ..typing import Batch, Observation, Action
from ..utils import pretty_repr, jit
from .._base.mixins import RandomStateMixin, CopyMixin


class Inputs(NamedTuple):
    args: Any
    static_argnums: Tuple[int, ...]

    def __repr__(self):
        return pretty_repr(self)


class ExampleData(NamedTuple):
    inputs: Inputs
    output: Any

    def __repr__(self):
        return pretty_repr(self)


class ArgsType1(NamedTuple):
    S: Batch[Observation]
    A: Batch[Action]
    is_training: bool

    def __repr__(self):
        return pretty_repr(self)


class ArgsType2(NamedTuple):
    S: Batch[Observation]
    is_training: bool

    def __repr__(self):
        return pretty_repr(self)


class ModelTypes(NamedTuple):
    type1: ArgsType1
    type2: ArgsType2

    def __repr__(self):
        return pretty_repr(self)


class BaseFunc(ABC, RandomStateMixin, CopyMixin):
    """ Abstract base class for function approximators: coax.V, coax.Q, coax.Policy """

    def __init__(self, func, observation_space, action_space=None, random_seed=None):

        if not isinstance(observation_space, Space):
            raise TypeError(
                f"observation_space must be derived from gym.Space, got: {type(observation_space)}")
        self.observation_space = observation_space

        if action_space is not None:
            if not isinstance(action_space, Space):
                raise TypeError(
                    f"action_space must be derived from gym.Space, got: {type(action_space)}")
            self.action_space = action_space

        self.random_seed = random_seed  # also initializes self.rng via RandomStateMixin
        self._jitted_funcs = {}

        # Haiku-transform the provided func
        example_data = self._check_signature(func)
        static_argnums = tuple(i + 3 for i in example_data.inputs.static_argnums)
        transformed = hk.transform_with_state(func)
        self._function = jit(transformed.apply, static_argnums=static_argnums)

        # init function params and state
        self._params, self._function_state = transformed.init(self.rng, *example_data.inputs.args)

        # check if output has the expected shape etc.
        output, _ = \
            self._function(self.params, self.function_state, self.rng, *example_data.inputs.args)
        self._check_output(output, example_data.output)

        def soft_update_func(old, new, tau):
            return jax.tree_multimap(lambda a, b: (1 - tau) * a + tau * b, old, new)

        self._soft_update_func = jit(soft_update_func)

    def soft_update(self, other, tau):
        r""" Synchronize the current instance with ``other`` through exponential smoothing:

        .. math::

            \theta\ \leftarrow\ \theta + \tau\, (\theta_\text{new} - \theta)

        Parameters
        ----------
        other

            A seperate copy of the current object. This object will hold the new parameters
            :math:`\theta_\text{new}`.

        tau : float between 0 and 1, optional

            If we set :math:`\tau=1` we do a hard update. If we pick a smaller value, we do a smooth
            update.

        """
        if not isinstance(other, self.__class__):
            raise TypeError("'self' and 'other' must be of the same type")

        self.params = self._soft_update_func(self.params, other.params, tau)
        self.function_state = self._soft_update_func(self.function_state, other.function_state, tau)

    @property
    def params(self):
        """ The parameters (weights) of the function approximator. """
        return self._params

    @params.setter
    def params(self, new_params):
        if jax.tree_structure(new_params) != jax.tree_structure(self._params):
            raise TypeError("new params must have the same structure as old params")
        self._params = new_params

    @property
    def function(self):
        r"""

        The function approximator itself, defined as a JIT-compiled pure function. This function may
        be called directly as:

        .. code:: python

            output, function_state = obj.function(obj.params, obj.function_state, obj.rng, *inputs)

        """
        return self._function

    @property
    def function_state(self):
        """ The state of the function approximator, see :func:`haiku.transform_with_state`. """
        return self._function_state

    @function_state.setter
    def function_state(self, new_function_state):
        if jax.tree_structure(new_function_state) != jax.tree_structure(self._function_state):
            raise TypeError("new function_state must have the same structure as old function_state")
        self._function_state = new_function_state

    @abstractmethod
    def _check_signature(self, func):
        """ Check if func has expected input signature; returns example_data; raises TypeError """

    @abstractmethod
    def _check_output(self, actual, expected):
        """ Check if func has expected output signature; raises TypeError """

    @abstractmethod
    def example_data(self, env, **kwargs):
        r"""

        A small utility function that generates example input and output data. These may be useful
        for writing and debugging your own custom function approximators.

        """
