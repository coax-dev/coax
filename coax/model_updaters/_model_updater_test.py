from copy import deepcopy

from optax import sgd

from .._base.test_case import TestCase
from .._core.stochastic_transition_model import StochasticTransitionModel
from ..utils import get_transition_batch
from ..regularizers import EntropyRegularizer
from ._model_updater import ModelUpdater


class TestModelUpdater(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition_batch(self.env_discrete, random_seed=42)
        self.transition_boxspace = get_transition_batch(self.env_boxspace, random_seed=42)

    def test_update_type1(self):
        env = self.env_discrete
        func_p = self.func_p_type1
        transition_batch = self.transition_discrete

        p = StochasticTransitionModel(func_p, env, random_seed=11)
        updater = ModelUpdater(p, optimizer=sgd(1.0))

        params = deepcopy(p.params)
        function_state = deepcopy(p.function_state)

        updater.update(transition_batch)

        self.assertPytreeNotEqual(params, p.params)
        self.assertPytreeNotEqual(function_state, p.function_state)

    def test_update_type2(self):
        env = self.env_discrete
        func_p = self.func_p_type2
        transition_batch = self.transition_discrete

        p = StochasticTransitionModel(func_p, env, random_seed=11)
        updater = ModelUpdater(p, optimizer=sgd(1.0))

        params = deepcopy(p.params)
        function_state = deepcopy(p.function_state)

        updater.update(transition_batch)

        self.assertPytreeNotEqual(params, p.params)
        self.assertPytreeNotEqual(function_state, p.function_state)

    def test_policyreg(self):
        env = self.env_discrete
        func_p = self.func_p_type1
        transition_batch = self.transition_discrete

        p = StochasticTransitionModel(func_p, env, random_seed=11)

        params_init = deepcopy(p.params)
        function_state_init = deepcopy(p.function_state)

        # first update without policy regularizer
        updater = ModelUpdater(p, optimizer=sgd(1.0))
        updater.update(transition_batch)
        params_without_reg = deepcopy(p.params)
        function_state_without_reg = deepcopy(p.function_state)
        self.assertPytreeNotEqual(params_without_reg, params_init)
        self.assertPytreeNotEqual(function_state_without_reg, function_state_init)

        # reset weights
        p = StochasticTransitionModel(func_p, env, random_seed=11)
        self.assertPytreeAlmostEqual(params_init, p.params)
        self.assertPytreeAlmostEqual(function_state_init, p.function_state)

        # then update with policy regularizer
        reg = EntropyRegularizer(p, beta=1.0)
        updater = ModelUpdater(p, optimizer=sgd(1.0), regularizer=reg)
        updater.update(transition_batch)
        params_with_reg = deepcopy(p.params)
        function_state_with_reg = deepcopy(p.function_state)
        self.assertPytreeNotEqual(params_with_reg, params_init)
        self.assertPytreeNotEqual(params_with_reg, params_without_reg)  # <---- important
        self.assertPytreeNotEqual(function_state_with_reg, function_state_init)
        self.assertPytreeAlmostEqual(function_state_with_reg, function_state_without_reg)  # same!
