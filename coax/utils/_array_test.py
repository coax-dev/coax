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

import jax
import jax.numpy as jnp
from haiku import PRNGSequence

from .._base.test_case import TestCase
from ._array import argmax


class TestArrayUtils(TestCase):

    def test_argmax_consistent(self):
        rngs = PRNGSequence(13)

        vec = jax.random.normal(next(rngs), shape=(5,))
        mat = jax.random.normal(next(rngs), shape=(3, 5))
        ten = jax.random.normal(next(rngs), shape=(3, 5, 7))

        self.assertEqual(
            argmax(next(rngs), vec), jnp.argmax(vec, axis=-1))
        self.assertArrayAlmostEqual(
            argmax(next(rngs), mat), jnp.argmax(mat, axis=-1))
        self.assertArrayAlmostEqual(
            argmax(next(rngs), mat, axis=0), jnp.argmax(mat, axis=0))
        self.assertArrayAlmostEqual(
            argmax(next(rngs), ten), jnp.argmax(ten, axis=-1))
        self.assertArrayAlmostEqual(
            argmax(next(rngs), ten, axis=0), jnp.argmax(ten, axis=0))
        self.assertArrayAlmostEqual(
            argmax(next(rngs), ten, axis=1), jnp.argmax(ten, axis=1))

    def test_argmax_random_tiebreaking(self):
        rngs = PRNGSequence(13)

        vec = jnp.ones(shape=(5,))
        mat = jnp.ones(shape=(3, 5))

        self.assertEqual(argmax(next(rngs), vec), 2)  # not zero
        self.assertArrayAlmostEqual(argmax(next(rngs), mat), [1, 1, 3])
