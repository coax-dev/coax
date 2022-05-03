import gym
import jax
import jax.numpy as jnp
import numpy as onp
from haiku import PRNGSequence

from .._base.test_case import TestCase
from ..proba_dists import NormalDist
from ._array import (
    argmax,
    check_preprocessors,
    chunks_pow2,
    default_preprocessor,
    get_transition_batch,
    tree_sample,
)


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

    def test_check_preprocessors(self):
        box = gym.spaces.Box(low=onp.finfo('float32').min, high=onp.finfo('float32').max, shape=[7])
        p0 = NormalDist(box).preprocess_variate
        p1 = default_preprocessor(box)

        def p2(rng, x):
            return 'garbage'

        msg = r"need at least two preprocessors in order to run test"
        with self.assertRaisesRegex(ValueError, msg):
            check_preprocessors(box)
        with self.assertRaisesRegex(ValueError, msg):
            check_preprocessors(box, p0)

        self.assertTrue(check_preprocessors(box, p0, p0, p0))
        self.assertFalse(check_preprocessors(box, p0, p1))
        self.assertFalse(check_preprocessors(box, p0, p2))
        self.assertFalse(check_preprocessors(box, p1, p2))

    def test_default_preprocessor(self):
        rngs = PRNGSequence(13)

        box = gym.spaces.Box(low=0, high=1, shape=(2, 3))
        dsc = gym.spaces.Discrete(7)
        mbn = gym.spaces.MultiBinary(11)
        mds = gym.spaces.MultiDiscrete(nvec=[3, 5])
        tup = gym.spaces.Tuple((box, dsc, mbn, mds))
        dct = gym.spaces.Dict({'box': box, 'dsc': dsc, 'mbn': mbn, 'mds': mds})

        self.assertArrayShape(default_preprocessor(box)(next(rngs), box.sample()), (1, 2, 3))
        self.assertArrayShape(default_preprocessor(dsc)(next(rngs), dsc.sample()), (1, 7))
        self.assertArrayShape(default_preprocessor(mbn)(next(rngs), mbn.sample()), (1, 11))
        self.assertArrayShape(default_preprocessor(mds)(next(rngs), mds.sample())[0], (1, 3))
        self.assertArrayShape(default_preprocessor(mds)(next(rngs), mds.sample())[1], (1, 5))

        self.assertArrayShape(default_preprocessor(tup)(next(rngs), tup.sample())[0], (1, 2, 3))
        self.assertArrayShape(default_preprocessor(tup)(next(rngs), tup.sample())[1], (1, 7))
        self.assertArrayShape(default_preprocessor(tup)(next(rngs), tup.sample())[2], (1, 11))
        self.assertArrayShape(default_preprocessor(tup)(next(rngs), tup.sample())[3][0], (1, 3))
        self.assertArrayShape(default_preprocessor(tup)(next(rngs), tup.sample())[3][1], (1, 5))

        self.assertArrayShape(default_preprocessor(dct)(next(rngs), dct.sample())['box'], (1, 2, 3))
        self.assertArrayShape(default_preprocessor(dct)(next(rngs), dct.sample())['dsc'], (1, 7))
        self.assertArrayShape(default_preprocessor(dct)(next(rngs), dct.sample())['mbn'], (1, 11))
        self.assertArrayShape(default_preprocessor(dct)(next(rngs), dct.sample())['mds'][0], (1, 3))
        self.assertArrayShape(default_preprocessor(dct)(next(rngs), dct.sample())['mds'][1], (1, 5))

        mds_batch = jax.tree_multimap(lambda *x: jnp.stack(x), *(mds.sample() for _ in range(7)))
        self.assertArrayShape(default_preprocessor(mds)(next(rngs), mds_batch)[0], (7, 3))
        self.assertArrayShape(default_preprocessor(mds)(next(rngs), mds_batch)[1], (7, 5))

    def test_chunks_pow2(self):
        chunk_sizes = (2048, 1024, 512, 64, 32, 1)
        tn = get_transition_batch(self.env_discrete, batch_size=sum(chunk_sizes))

        for chunk, chunk_size in zip(chunks_pow2(tn), chunk_sizes):
            self.assertEqual(chunk.batch_size, chunk_size)

    def test_tree_sample(self):
        rngs = PRNGSequence(42)
        tn = get_transition_batch(self.env_discrete, batch_size=5)

        tn_sample = tree_sample(tn, next(rngs), n=3)
        assert tn_sample.batch_size == 3

        tn_sample = tree_sample(tn, next(rngs), n=7, replace=True)
        assert tn_sample.batch_size == 7

        msg = r"Cannot take a larger sample than population when 'replace=False'"
        with self.assertRaisesRegex(ValueError, msg):
            tree_sample(tn, next(rngs), n=7, replace=False)
