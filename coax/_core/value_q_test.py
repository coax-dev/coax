# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

import numpy as onp
import jax
import jax.numpy as jnp

from .._base.test_case import TestCase, DummyFuncApprox
from ..utils import get_transition
from .value_q import Q


class TestQ(TestCase):

    def setUp(self):
        self.funcs = {
            'discrete': DummyFuncApprox(self.env_discrete),
            'box': DummyFuncApprox(self.env_box)}
        self.transitions = {
            'discrete': get_transition(self.env_discrete),
            'box': get_transition(self.env_box)}
        self.transitions_batch = {
            'discrete': get_transition(self.env_discrete).to_batch(),
            'box': get_transition(self.env_box).to_batch()}

    def tearDown(self):
        del self.funcs, self.transitions, self.transitions_batch

    def test_qtype2_nondiscrete(self):
        # cannot define a type-II q-function on a non-discrete action space
        msg = ("type-II q-function is not (yet) implemented for "
               "non-discrete action spaces")
        with self.assertRaises(NotImplementedError, msg=msg):
            Q(self.funcs['box'], qtype=2)

        # this should be fine
        Q(self.funcs['box'], qtype=1)
        Q(self.funcs['discrete'], qtype=1)
        Q(self.funcs['discrete'], qtype=2)

    def test_batch_eval_type1_discrete(self):
        S = self.transitions_batch['discrete'].S
        A = self.transitions_batch['discrete'].A
        q = Q(self.funcs['discrete'], qtype=1)

        # without A
        Q_s = q.batch_eval(S)
        self.assertArrayShape(Q_s, (1, self.env_discrete.action_space.n))
        self.assertArraySubdtypeFloat(Q_s)

        # with A
        Q_sa = q.batch_eval(S, A)
        self.assertArrayShape(Q_sa, (1,))
        self.assertArraySubdtypeFloat(Q_sa)
        self.assertArrayAlmostEqual(Q_sa, Q_s[:, A[0]])

    def test_batch_eval_type2_discrete(self):
        S = self.transitions_batch['discrete'].S
        A = self.transitions_batch['discrete'].A
        q = Q(self.funcs['discrete'], qtype=2)

        # without A
        Q_s = q.batch_eval(S)
        self.assertArrayShape(Q_s, (1, self.env_discrete.action_space.n))
        self.assertArraySubdtypeFloat(Q_s)

        # with A
        Q_sa = q.batch_eval(S, A)
        self.assertArrayShape(Q_sa, (1,))
        self.assertArraySubdtypeFloat(Q_sa)
        self.assertArrayAlmostEqual(Q_sa, Q_s[:, A[0]])

    def test_batch_eval_type1_box(self):
        S = self.transitions_batch['box'].S
        A = self.transitions_batch['box'].A
        q = Q(self.funcs['box'], qtype=1)

        # type-I requires A
        msg = "input 'A' is required for type-I q-function when action space is non-discrete"
        with self.assertRaises(ValueError, msg=msg):
            q.batch_eval(S)

        Q_sa = q.batch_eval(S, A)
        self.assertArrayShape(Q_sa, (1,))
        self.assertArraySubdtypeFloat(Q_sa)

    def test_call_type1_discrete(self):
        s = self.transitions['discrete'].s
        a = self.transitions['discrete'].a
        q = Q(self.funcs['discrete'], qtype=1)

        # without a
        q_s = q(s)
        self.assertArrayShape(q_s, (self.env_discrete.action_space.n,))
        self.assertArraySubdtypeFloat(q_s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_s)
        self.assertArrayAlmostEqual(q_sa, q_s[a])

    def test_call_type2_discrete(self):
        s = self.transitions['discrete'].s
        a = self.transitions['discrete'].a
        q = Q(self.funcs['discrete'], qtype=2)

        # without a
        q_s = q(s)
        self.assertArrayShape(q_s, (self.env_discrete.action_space.n,))
        self.assertArraySubdtypeFloat(q_s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_s)
        self.assertArrayAlmostEqual(q_sa, q_s[a])

    def test_call_type1_box(self):
        s = self.transitions['box'].s
        a = self.transitions['box'].a
        q = Q(self.funcs['box'], qtype=1)

        # type-I requires a if actions space is non-discrete
        msg = "input 'A' is required for type-I q-function when action space is non-discrete"
        with self.assertRaises(ValueError, msg=msg):
            q(s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_sa)

    def test_apply_q1_as_q2(self):
        n = 3  # num_actions
        q = Q(self.funcs['discrete'], qtype=1)

        def q1_func(params, state, rng, S, A, is_training):
            return jnp.array([encode(s, a) for s, a in zip(S, A)]), state

        def encode(s, a):
            return 2 ** a + 2 ** (s + n)

        def decode(i):
            b = onp.array(list(bin(i)))[::-1]
            a = onp.argwhere(b[:n] == '1').item()
            s = onp.argwhere(b[n:] == '1').item()
            return s, a

        q._apply_func = q1_func
        rng = jax.random.PRNGKey(0)
        params = {'body': (), 'action_processor': (), 'state_action_combiner': (), 'head_q1': ()}
        state = {'body': (), 'action_processor': (), 'state_action_combiner': (), 'head_q1': ()}
        is_training = True

        S = jnp.array([5, 7, 11, 13, 17, 19, 23])
        encoded_rows, _ = q.apply_func_type2(params, state, rng, S, is_training)
        for s, encoded_row in zip(S, encoded_rows):
            for a, x in enumerate(encoded_row):
                s_, a_ = decode(x)
                self.assertEqual(s_, s)
                self.assertEqual(a_, a)

    def test_apply_q2_as_q1(self):
        n = 3  # num_actions
        q = Q(self.funcs['discrete'], qtype=2)

        def q2_func(params, state, rng, S, is_training):
            return jnp.tile(jnp.arange(n), reps=(S.shape[0], 1)), state

        q._apply_func = q2_func
        rng = jax.random.PRNGKey(0)
        params = {'body': (), 'head_q2': ()}
        state = {'body': (), 'head_q2': ()}
        is_training = True

        S = jnp.array([5, 7, 11, 13, 17, 19, 23])
        A = jnp.array([2, 0, 1, 1, 0, 1, 2])
        Q_sa, _ = q.apply_func_type1(params, state, rng, S, A, is_training)
        self.assertArrayAlmostEqual(Q_sa, A)
