from copy import deepcopy

from optax import sgd

from .._base.test_case import TestCase
from .._core.q import Q
from .._core.stochastic_q import StochasticQ
from .._core.policy import Policy
from ..utils import get_transition_batch
from ._clippeddoubleqlearning import ClippedDoubleQLearning


class TestClippedDoubleQLearning(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition_batch(self.env_discrete, random_seed=42)
        self.transition_boxspace = get_transition_batch(self.env_boxspace, random_seed=42)

    def test_update_discrete_type1(self):
        env = self.env_discrete
        func_q = self.func_q_type1
        transition_batch = self.transition_discrete

        q1 = Q(func_q, env)
        q2 = Q(func_q, env)
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()
        updater1 = ClippedDoubleQLearning(q1, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))
        updater2 = ClippedDoubleQLearning(q2, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

        params1 = deepcopy(q1.params)
        params2 = deepcopy(q2.params)
        function_state1 = deepcopy(q1.function_state)
        function_state2 = deepcopy(q2.function_state)

        updater1.update(transition_batch)
        updater2.update(transition_batch)

        self.assertPytreeNotEqual(params1, q1.params)
        self.assertPytreeNotEqual(params2, q2.params)
        self.assertPytreeNotEqual(function_state1, q1.function_state)
        self.assertPytreeNotEqual(function_state2, q2.function_state)

    def test_update_discrete_stochastic_type1(self):
        env = self.env_discrete
        func_q = self.func_q_stochastic_type1
        transition_batch = self.transition_discrete

        q1 = StochasticQ(func_q, env, value_range=(0, 1))
        q2 = StochasticQ(func_q, env, value_range=(0, 1))
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()
        updater1 = ClippedDoubleQLearning(q1, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))
        updater2 = ClippedDoubleQLearning(q2, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

        params1 = deepcopy(q1.params)
        params2 = deepcopy(q2.params)
        function_state1 = deepcopy(q1.function_state)
        function_state2 = deepcopy(q2.function_state)

        updater1.update(transition_batch)
        updater2.update(transition_batch)

        self.assertPytreeNotEqual(params1, q1.params)
        self.assertPytreeNotEqual(params2, q2.params)
        self.assertPytreeNotEqual(function_state1, q1.function_state)
        self.assertPytreeNotEqual(function_state2, q2.function_state)

    def test_update_discrete_type2(self):
        env = self.env_discrete
        func_q = self.func_q_type2
        transition_batch = self.transition_discrete

        q1 = Q(func_q, env)
        q2 = Q(func_q, env)
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()
        updater1 = ClippedDoubleQLearning(q1, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))
        updater2 = ClippedDoubleQLearning(q2, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

        params1 = deepcopy(q1.params)
        params2 = deepcopy(q2.params)
        function_state1 = deepcopy(q1.function_state)
        function_state2 = deepcopy(q2.function_state)

        updater1.update(transition_batch)
        updater2.update(transition_batch)

        self.assertPytreeNotEqual(params1, q1.params)
        self.assertPytreeNotEqual(params2, q2.params)
        self.assertPytreeNotEqual(function_state1, q1.function_state)
        self.assertPytreeNotEqual(function_state2, q2.function_state)

    def test_update_discrete_stochastic_type2(self):
        env = self.env_discrete
        func_q = self.func_q_stochastic_type2
        transition_batch = self.transition_discrete

        q1 = StochasticQ(func_q, env, value_range=(0, 1))
        q2 = StochasticQ(func_q, env, value_range=(0, 1))
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()
        updater1 = ClippedDoubleQLearning(q1, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))
        updater2 = ClippedDoubleQLearning(q2, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

        params1 = deepcopy(q1.params)
        params2 = deepcopy(q2.params)
        function_state1 = deepcopy(q1.function_state)
        function_state2 = deepcopy(q2.function_state)

        updater1.update(transition_batch)
        updater2.update(transition_batch)

        self.assertPytreeNotEqual(params1, q1.params)
        self.assertPytreeNotEqual(params2, q2.params)
        self.assertPytreeNotEqual(function_state1, q1.function_state)
        self.assertPytreeNotEqual(function_state2, q2.function_state)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func_q = self.func_q_type1
        func_pi = self.func_pi_boxspace
        transition_batch = self.transition_boxspace

        q1 = Q(func_q, env)
        q2 = Q(func_q, env)
        pi1 = Policy(func_pi, env)
        pi2 = Policy(func_pi, env)
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()
        updater1 = ClippedDoubleQLearning(
            q1, pi_targ_list=[pi1, pi2], q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))
        updater2 = ClippedDoubleQLearning(
            q2, pi_targ_list=[pi1, pi2], q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

        params1 = deepcopy(q1.params)
        params2 = deepcopy(q2.params)
        function_state1 = deepcopy(q1.function_state)
        function_state2 = deepcopy(q2.function_state)

        updater1.update(transition_batch)
        updater2.update(transition_batch)

        self.assertPytreeNotEqual(params1, q1.params)
        self.assertPytreeNotEqual(params2, q2.params)
        self.assertPytreeNotEqual(function_state1, q1.function_state)
        self.assertPytreeNotEqual(function_state2, q2.function_state)

    def test_update_boxspace_stochastic(self):
        env = self.env_boxspace
        func_q = self.func_q_stochastic_type1
        func_pi = self.func_pi_boxspace
        transition_batch = self.transition_boxspace

        q1 = StochasticQ(func_q, env, value_range=(0, 1))
        q2 = StochasticQ(func_q, env, value_range=(0, 1))
        pi1 = Policy(func_pi, env)
        pi2 = Policy(func_pi, env)
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()
        updater1 = ClippedDoubleQLearning(
            q1, pi_targ_list=[pi1, pi2], q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))
        updater2 = ClippedDoubleQLearning(
            q2, pi_targ_list=[pi1, pi2], q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

        params1 = deepcopy(q1.params)
        params2 = deepcopy(q2.params)
        function_state1 = deepcopy(q1.function_state)
        function_state2 = deepcopy(q2.function_state)

        updater1.update(transition_batch)
        updater2.update(transition_batch)

        self.assertPytreeNotEqual(params1, q1.params)
        self.assertPytreeNotEqual(params2, q2.params)
        self.assertPytreeNotEqual(function_state1, q1.function_state)
        self.assertPytreeNotEqual(function_state2, q2.function_state)

    def test_discrete_with_pi(self):
        env = self.env_discrete
        func_q = self.func_q_type1
        func_pi = self.func_pi_discrete

        q1 = Q(func_q, env)
        q2 = Q(func_q, env)
        pi1 = Policy(func_pi, env)
        pi2 = Policy(func_pi, env)
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()

        msg = r"pi_targ_list is ignored, because action space is discrete"
        with self.assertWarnsRegex(UserWarning, msg):
            ClippedDoubleQLearning(
                q1, pi_targ_list=[pi1, pi2], q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

    def test_boxspace_without_pi(self):
        env = self.env_boxspace
        func_q = self.func_q_type1

        q1 = Q(func_q, env)
        q2 = Q(func_q, env)
        q_targ1 = q1.copy()
        q_targ2 = q2.copy()

        msg = r"pi_targ_list must be provided if action space is not discrete"
        with self.assertRaisesRegex(TypeError, msg):
            ClippedDoubleQLearning(q1, q_targ_list=[q_targ1, q_targ2], optimizer=sgd(1.0))

    def test_update_discrete_nogrid(self):
        env = self.env_discrete
        func_q = self.func_q_type1

        q = Q(func_q, env)
        q_targ = q.copy()

        msg = r"len\(q_targ_list\) must be at least 2"
        with self.assertRaisesRegex(ValueError, msg):
            ClippedDoubleQLearning(q, q_targ_list=[q_targ], optimizer=sgd(1.0))

    def test_update_boxspace_nogrid(self):
        env = self.env_boxspace
        func_q = self.func_q_type1
        func_pi = self.func_pi_boxspace

        q = Q(func_q, env)
        pi = Policy(func_pi, env)
        q_targ = q.copy()

        msg = r"len\(q_targ_list\) \* len\(pi_targ_list\) must be at least 2"
        with self.assertRaisesRegex(ValueError, msg):
            ClippedDoubleQLearning(q, pi_targ_list=[pi], q_targ_list=[q_targ], optimizer=sgd(1.0))
