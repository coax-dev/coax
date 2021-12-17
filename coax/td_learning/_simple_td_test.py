from copy import deepcopy

from optax import sgd

from .._base.test_case import TestCase
from .._core.v import V
from .._core.policy import Policy
from ..utils import get_transition_batch
from ..regularizers import EntropyRegularizer
from ..value_transforms import LogTransform
from ._simple_td import SimpleTD


class TestSimpleTD(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition_batch(self.env_discrete, random_seed=42)
        self.transition_boxspace = get_transition_batch(self.env_boxspace, random_seed=42)

    def test_update_discrete(self):
        env = self.env_discrete
        func_v = self.func_v

        v = V(func_v, env, random_seed=11)
        v_targ = v.copy()
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0))

        params = deepcopy(v.params)
        function_state = deepcopy(v.function_state)

        updater.update(self.transition_discrete)

        self.assertPytreeNotEqual(params, v.params)
        self.assertPytreeNotEqual(function_state, v.function_state)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func_v = self.func_v

        v = V(func_v, env, random_seed=11)
        v_targ = v.copy()
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0))

        params = deepcopy(v.params)
        function_state = deepcopy(v.function_state)

        updater.update(self.transition_boxspace)

        self.assertPytreeNotEqual(params, v.params)
        self.assertPytreeNotEqual(function_state, v.function_state)

    def test_policyreg_discrete(self):
        env = self.env_discrete
        func_v = self.func_v
        func_pi = self.func_pi_discrete
        transition_batch = self.transition_discrete

        v = V(func_v, env, random_seed=11)
        pi = Policy(func_pi, env, random_seed=17)
        v_targ = v.copy()

        params_init = deepcopy(v.params)
        function_state_init = deepcopy(v.function_state)

        # first update without policy regularizer
        policy_reg = EntropyRegularizer(pi, beta=1.0)
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0))
        updater.update(transition_batch)
        params_without_reg = deepcopy(v.params)
        function_state_without_reg = deepcopy(v.function_state)
        self.assertPytreeNotEqual(params_without_reg, params_init)
        self.assertPytreeNotEqual(function_state_without_reg, function_state_init)

        # reset weights
        v = V(func_v, env, random_seed=11)
        pi = Policy(func_pi, env, random_seed=17)
        v_targ = v.copy()
        self.assertPytreeAlmostEqual(params_init, v.params)
        self.assertPytreeAlmostEqual(function_state_init, v.function_state)

        # then update with policy regularizer
        policy_reg = EntropyRegularizer(pi, beta=1.0)
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0), policy_regularizer=policy_reg)
        updater.update(transition_batch)
        params_with_reg = deepcopy(v.params)
        function_state_with_reg = deepcopy(v.function_state)
        self.assertPytreeNotEqual(params_with_reg, params_init)
        self.assertPytreeNotEqual(function_state_with_reg, function_state_init)
        self.assertPytreeNotEqual(params_with_reg, params_without_reg)
        self.assertPytreeAlmostEqual(function_state_with_reg, function_state_without_reg)  # same!

    def test_policyreg_boxspace(self):
        env = self.env_boxspace
        func_v = self.func_v
        func_pi = self.func_pi_boxspace
        transition_batch = self.transition_boxspace

        v = V(func_v, env, random_seed=11)
        pi = Policy(func_pi, env, random_seed=17)
        v_targ = v.copy()

        params_init = deepcopy(v.params)
        function_state_init = deepcopy(v.function_state)

        # first update without policy regularizer
        policy_reg = EntropyRegularizer(pi, beta=1.0)
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0))
        updater.update(transition_batch)
        params_without_reg = deepcopy(v.params)
        function_state_without_reg = deepcopy(v.function_state)
        self.assertPytreeNotEqual(params_without_reg, params_init)
        self.assertPytreeNotEqual(function_state_without_reg, function_state_init)

        # reset weights
        v = V(func_v, env, random_seed=11)
        pi = Policy(func_pi, env, random_seed=17)
        v_targ = v.copy()
        self.assertPytreeAlmostEqual(params_init, v.params)
        self.assertPytreeAlmostEqual(function_state_init, v.function_state)

        # then update with policy regularizer
        policy_reg = EntropyRegularizer(pi, beta=1.0)
        updater = SimpleTD(v, v_targ, optimizer=sgd(1.0), policy_regularizer=policy_reg)
        updater.update(transition_batch)
        params_with_reg = deepcopy(v.params)
        function_state_with_reg = deepcopy(v.function_state)
        self.assertPytreeNotEqual(params_with_reg, params_init)
        self.assertPytreeNotEqual(function_state_with_reg, function_state_init)
        self.assertPytreeNotEqual(params_with_reg, params_without_reg)
        self.assertPytreeAlmostEqual(function_state_with_reg, function_state_without_reg)  # same!

    def test_value_transform(self):
        env = self.env_discrete
        func_v = self.func_v
        transition_batch = self.transition_discrete

        v = V(func_v, env, random_seed=11)

        params_init = deepcopy(v.params)
        function_state_init = deepcopy(v.function_state)

        # first update without value transform
        updater = SimpleTD(v, optimizer=sgd(1.0))
        updater.update(transition_batch)
        params_without_reg = deepcopy(v.params)
        function_state_without_reg = deepcopy(v.function_state)
        self.assertPytreeNotEqual(params_without_reg, params_init)
        self.assertPytreeNotEqual(function_state_without_reg, function_state_init)

        # reset weights
        v = V(func_v, env, value_transform=LogTransform(), random_seed=11)
        self.assertPytreeAlmostEqual(params_init, v.params)
        self.assertPytreeAlmostEqual(function_state_init, v.function_state)

        # then update with value transform
        updater = SimpleTD(v, optimizer=sgd(1.0))
        updater.update(transition_batch)
        params_with_reg = deepcopy(v.params)
        function_state_with_reg = deepcopy(v.function_state)
        self.assertPytreeNotEqual(params_with_reg, params_init)
        self.assertPytreeNotEqual(function_state_with_reg, function_state_init)
        self.assertPytreeNotEqual(params_with_reg, params_without_reg)
        self.assertPytreeAlmostEqual(function_state_with_reg, function_state_without_reg)  # same!
