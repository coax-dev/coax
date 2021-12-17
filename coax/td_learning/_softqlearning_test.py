from copy import deepcopy

from optax import sgd

from .._base.test_case import TestCase
from .._core.q import Q
from .._core.policy import Policy
from ..utils import get_transition_batch
from ..regularizers import EntropyRegularizer
from ._softqlearning import SoftQLearning


class TestSoftQLearning(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition_batch(self.env_discrete, random_seed=42)
        self.transition_boxspace = get_transition_batch(self.env_boxspace, random_seed=42)

    def test_update_discrete_type1(self):
        env = self.env_discrete
        func_q = self.func_q_type1
        transition_batch = self.transition_discrete

        q = Q(func_q, env, random_seed=11)
        q_targ = q.copy()
        updater = SoftQLearning(q, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(transition_batch)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_discrete_type2(self):
        env = self.env_discrete
        func_q = self.func_q_type2
        transition_batch = self.transition_discrete

        q = Q(func_q, env, random_seed=11)
        q_targ = q.copy()
        updater = SoftQLearning(q, q_targ, optimizer=sgd(1.0))

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(transition_batch)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func_q = self.func_q_type1

        q = Q(func_q, env, random_seed=11)
        q_targ = q.copy()

        msg = r"SoftQLearning class is only implemented for discrete actions spaces"
        with self.assertRaisesRegex(NotImplementedError, msg):
            SoftQLearning(q, q_targ, optimizer=sgd(1.0))

    def test_policyreg_discrete(self):
        env = self.env_discrete
        func_q = self.func_q_type1
        func_pi = self.func_pi_discrete
        transition_batch = self.transition_discrete

        q = Q(func_q, env, random_seed=11)
        pi = Policy(func_pi, env, random_seed=17)
        q_targ = q.copy()

        params_init = deepcopy(q.params)
        function_state_init = deepcopy(q.function_state)

        # first update without policy regularizer
        policy_reg = EntropyRegularizer(pi, beta=1.0)
        updater = SoftQLearning(q, q_targ, optimizer=sgd(1.0))
        updater.update(transition_batch)
        params_without_reg = deepcopy(q.params)
        function_state_without_reg = deepcopy(q.function_state)
        self.assertPytreeNotEqual(params_without_reg, params_init)
        self.assertPytreeNotEqual(function_state_without_reg, function_state_init)

        # reset weights
        q = Q(func_q, env, random_seed=11)
        pi = Policy(func_pi, env, random_seed=17)
        q_targ = q.copy()
        self.assertPytreeAlmostEqual(params_init, q.params)
        self.assertPytreeAlmostEqual(function_state_init, q.function_state)

        # then update with policy regularizer
        policy_reg = EntropyRegularizer(pi, beta=1.0)
        updater = SoftQLearning(q, q_targ, optimizer=sgd(1.0), policy_regularizer=policy_reg)
        print('updater.target_params:', updater.target_params)
        print('updater.target_function_state:', updater.target_function_state)
        updater.update(transition_batch)
        params_with_reg = deepcopy(q.params)
        function_state_with_reg = deepcopy(q.function_state)
        self.assertPytreeNotEqual(params_with_reg, params_init)
        self.assertPytreeNotEqual(function_state_with_reg, function_state_init)
        self.assertPytreeNotEqual(params_with_reg, params_without_reg)
        self.assertPytreeAlmostEqual(function_state_with_reg, function_state_without_reg)  # same!
