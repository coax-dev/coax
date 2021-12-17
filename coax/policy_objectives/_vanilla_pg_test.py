from copy import deepcopy

import jax.numpy as jnp
from optax import sgd

from .._base.test_case import TestCase
from .._core.policy import Policy
from ..utils import tree_ravel
from ..regularizers import EntropyRegularizer, KLDivRegularizer
from ._vanilla_pg import VanillaPG


class TestVanillaPG(TestCase):

    def test_update_discrete(self):
        env = self.env_discrete
        func = self.func_pi_discrete
        transitions = self.transitions_discrete
        print(transitions)

        pi = Policy(func, env)
        updater = VanillaPG(pi, optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)

    def test_update_box(self):
        env = self.env_boxspace
        func = self.func_pi_boxspace
        transitions = self.transitions_boxspace
        print(transitions)

        pi = Policy(func, env)
        updater = VanillaPG(pi, optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)

    def test_update_discrete_entropyreg(self):
        env = self.env_discrete
        func = self.func_pi_discrete
        transitions = self.transitions_discrete
        reg = EntropyRegularizer
        print(transitions)

        pi = Policy(func, env)
        updater = VanillaPG(pi, regularizer=reg(pi), optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)

    def test_update_box_entropyreg(self):
        env = self.env_boxspace
        func = self.func_pi_boxspace
        transitions = self.transitions_boxspace
        reg = EntropyRegularizer
        print(transitions)

        pi = Policy(func, env)
        updater = VanillaPG(pi, regularizer=reg(pi), optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)

    def test_update_discrete_kldivreg(self):
        env = self.env_discrete
        func = self.func_pi_discrete
        transitions = self.transitions_discrete
        reg = KLDivRegularizer
        print(transitions)

        pi = Policy(func, env)
        updater = VanillaPG(pi, regularizer=reg(pi), optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)

    def test_update_box_kldivreg(self):
        env = self.env_boxspace
        func = self.func_pi_boxspace
        transitions = self.transitions_boxspace
        reg = KLDivRegularizer
        print(transitions)

        pi = Policy(func, env)
        updater = VanillaPG(pi, regularizer=reg(pi), optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)
