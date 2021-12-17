from copy import deepcopy

import jax.numpy as jnp
from optax import sgd

from .._base.test_case import TestCase
from .._core.policy import Policy
from .._core.q import Q
from ..utils import tree_ravel
from ._deterministic_pg import DeterministicPG


class TestDeterministicPG(TestCase):

    def test_update_discrete(self):
        env = self.env_discrete
        func = self.func_pi_discrete
        transitions = self.transitions_discrete
        print(transitions)

        pi = Policy(func, env)
        q_targ = Q(self.func_q_type1, env)
        updater = DeterministicPG(pi, q_targ, optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)

    def test_update_boxspace(self):
        env = self.env_boxspace
        func = self.func_pi_boxspace
        transitions = self.transitions_boxspace
        print(transitions)

        pi = Policy(func, env)
        q_targ = Q(self.func_q_type1, env)
        updater = DeterministicPG(pi, q_targ, optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)
