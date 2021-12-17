from copy import deepcopy

import jax.numpy as jnp
from optax import sgd

from .._base.test_case import TestCase
from .._core.policy import Policy
from ..utils import tree_ravel
from ._ppo_clip import PPOClip


class TestPPOClip(TestCase):

    def test_update_discrete(self):
        env = self.env_discrete
        func = self.func_pi_discrete
        transitions = self.transitions_discrete
        print(transitions)

        pi = Policy(func, env)
        updater = PPOClip(pi, optimizer=sgd(1.0))

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
        updater = PPOClip(pi, optimizer=sgd(1.0))

        params = deepcopy(pi.params)
        function_state = deepcopy(pi.function_state)

        grads, _, _ = updater.grads_and_metrics(transitions, Adv=transitions.Rn)
        self.assertGreater(jnp.max(jnp.abs(tree_ravel(grads))), 0.)

        updater.update(transitions, Adv=transitions.Rn)

        self.assertPytreeNotEqual(function_state, pi.function_state)
        self.assertPytreeNotEqual(params, pi.params)
