import gym
import jax
import jax.numpy as jnp
import haiku as hk

from .._base.test_case import TestCase
from ._normal import NormalDist
from ._categorical import CategoricalDist
from ._composite import ProbaDist, StructureType

discrete = gym.spaces.Discrete(7)
box = gym.spaces.Box(low=0, high=1, shape=(3, 5))
multidiscrete = gym.spaces.MultiDiscrete([11, 13, 17])
multibinary = gym.spaces.MultiBinary(17)
tuple_flat = gym.spaces.Tuple((discrete, box))
dict_flat = gym.spaces.Dict({'d': discrete, 'b': box})
tuple_nested = gym.spaces.Tuple((discrete, box, multidiscrete, tuple_flat, dict_flat))
dict_nested = gym.spaces.Dict(
    {'d': discrete, 'b': box, 'm': multidiscrete, 't': tuple_flat, 'n': dict_flat})


class TestProbaDist(TestCase):
    decimal = 5

    def setUp(self):
        self.rngs = hk.PRNGSequence(13)

    def tearDown(self):
        del self.rngs

    def test_discrete(self):
        dist = ProbaDist(discrete)
        self.assertEqual(dist._structure_type, StructureType.LEAF)
        self.assertIsInstance(dist._structure, CategoricalDist)

        sample = dist.sample(dist.default_priors, next(self.rngs))
        self.assertArrayShape(sample, (1, discrete.n))
        self.assertAlmostEqual(sample.sum(), 1)

        mode = dist.mode(dist.default_priors)
        self.assertArrayShape(mode, (1, discrete.n))
        self.assertAlmostEqual(mode.sum(), 1)

    def test_box(self):
        dist = ProbaDist(box)
        self.assertEqual(dist._structure_type, StructureType.LEAF)
        self.assertIsInstance(dist._structure, NormalDist)

        sample = dist.sample(dist.default_priors, next(self.rngs))
        self.assertArrayShape(sample, (1, *box.shape))

        mode = dist.mode(dist.default_priors)
        self.assertArrayShape(mode, (1, *box.shape))

    def test_multidiscrete(self):
        dist = ProbaDist(multidiscrete)
        self.assertEqual(dist._structure_type, StructureType.LIST)
        sample = dist.sample(dist.default_priors, next(self.rngs))
        mode = dist.mode(dist.default_priors)
        for i, subdist in enumerate(dist._structure):
            self.assertEqual(subdist._structure_type, StructureType.LEAF)
            self.assertIsInstance(subdist._structure, CategoricalDist)
            self.assertEqual(subdist._structure.space.n, multidiscrete.nvec[i])
            self.assertArrayShape(sample[i], (1, multidiscrete.nvec[i]))
            self.assertAlmostEqual(sample[i].sum(), 1)
            self.assertArrayShape(mode[i], (1, multidiscrete.nvec[i]))
            self.assertAlmostEqual(mode[i].sum(), 1)

    def test_multibinary(self):
        dist = ProbaDist(multibinary)
        self.assertEqual(dist._structure_type, StructureType.LIST)
        sample = dist.sample(dist.default_priors, next(self.rngs))
        for i, subdist in enumerate(dist._structure):
            self.assertEqual(subdist._structure_type, StructureType.LEAF)
            self.assertIsInstance(subdist._structure, CategoricalDist)
            self.assertEqual(subdist._structure.space.n, 2)
            self.assertArrayShape(sample[i], (1, 2))
            self.assertAlmostEqual(sample[i].sum(), 1)

    def test_tuple_flat(self):
        dist = ProbaDist(tuple_flat)
        self.assertEqual(dist._structure_type, StructureType.LIST)
        for subdist in dist._structure:
            self.assertEqual(subdist._structure_type, StructureType.LEAF)
        self.assertIsInstance(dist._structure[0]._structure, CategoricalDist)
        self.assertIsInstance(dist._structure[1]._structure, NormalDist)

        sample = dist.sample(dist.default_priors, next(self.rngs))
        self.assertArrayShape(sample[0], (1, discrete.n))
        self.assertAlmostEqual(sample[0].sum(), 1)
        self.assertArrayShape(sample[1], (1, *box.shape))

        mode = dist.mode(dist.default_priors)
        self.assertArrayShape(mode[0], (1, discrete.n))
        self.assertAlmostEqual(mode[0].sum(), 1)
        self.assertArrayShape(mode[1], (1, *box.shape))

    def test_dict_flat(self):
        dist = ProbaDist(dict_flat)
        self.assertEqual(dist._structure_type, StructureType.DICT)
        for subdist in dist._structure.values():
            self.assertEqual(subdist._structure_type, StructureType.LEAF)
        self.assertIsInstance(dist._structure['d']._structure, CategoricalDist)
        self.assertIsInstance(dist._structure['b']._structure, NormalDist)

        sample = dist.sample(dist.default_priors, next(self.rngs))
        self.assertArrayShape(sample['d'], (1, discrete.n))
        self.assertAlmostEqual(sample['d'].sum(), 1)
        self.assertArrayShape(sample['b'], (1, *box.shape))

        mode = dist.mode(dist.default_priors)
        self.assertArrayShape(mode['d'], (1, discrete.n))
        self.assertAlmostEqual(mode['d'].sum(), 1)
        self.assertArrayShape(mode['b'], (1, *box.shape))

    def test_tuple_nested(self):
        dist = ProbaDist(tuple_nested)
        self.assertEqual(len(dist._structure), 5)
        self.assertEqual(dist._structure_type, StructureType.LIST)

        self.assertEqual(dist._structure[2]._structure_type, StructureType.LIST)
        self.assertIsInstance(dist._structure[2]._structure, list)
        self.assertIsInstance(dist._structure[2]._structure[1]._structure, CategoricalDist)
        self.assertEqual(dist._structure[3]._structure_type, StructureType.LIST)
        self.assertIsInstance(dist._structure[3]._structure, list)
        self.assertIsInstance(dist._structure[3]._structure[1]._structure, NormalDist)
        self.assertEqual(dist._structure[4]._structure_type, StructureType.DICT)
        self.assertIsInstance(dist._structure[4]._structure, dict)
        self.assertIsInstance(dist._structure[4]._structure['b']._structure, NormalDist)

        sample = dist.sample(dist.default_priors, next(self.rngs))
        self.assertEqual(len(sample), len(dist._structure))
        self.assertArrayShape(sample[3][0], (1, discrete.n))
        self.assertAlmostEqual(sample[3][0].sum(), 1)
        self.assertArrayShape(sample[3][1], (1, *box.shape))
        self.assertArrayShape(sample[4]['d'], (1, discrete.n))
        self.assertAlmostEqual(sample[4]['d'].sum(), 1)
        self.assertArrayShape(sample[4]['b'], (1, *box.shape))

        mode = dist.mode(dist.default_priors)
        self.assertEqual(len(mode), len(dist._structure))
        self.assertArrayShape(mode[3][0], (1, discrete.n))
        self.assertAlmostEqual(mode[3][0].sum(), 1)
        self.assertArrayShape(mode[3][1], (1, *box.shape))
        self.assertArrayShape(mode[4]['d'], (1, discrete.n))
        self.assertAlmostEqual(mode[4]['d'].sum(), 1)
        self.assertArrayShape(mode[4]['b'], (1, *box.shape))

    def test_dict_nested(self):
        dist = ProbaDist(dict_nested)
        self.assertEqual(len(dist._structure), 5)
        self.assertEqual(dist._structure_type, StructureType.DICT)

        self.assertEqual(dist._structure['m']._structure_type, StructureType.LIST)
        self.assertIsInstance(dist._structure['m']._structure, list)
        self.assertIsInstance(dist._structure['m']._structure[1]._structure, CategoricalDist)
        self.assertEqual(dist._structure['t']._structure_type, StructureType.LIST)
        self.assertIsInstance(dist._structure['t']._structure, list)
        self.assertIsInstance(dist._structure['t']._structure[1]._structure, NormalDist)
        self.assertEqual(dist._structure['n']._structure_type, StructureType.DICT)
        self.assertIsInstance(dist._structure['n']._structure, dict)
        self.assertIsInstance(dist._structure['n']._structure['b']._structure, NormalDist)

        sample = dist.sample(dist.default_priors, next(self.rngs))
        self.assertEqual(sample.keys(), dist._structure.keys())
        self.assertArrayShape(sample['t'][0], (1, discrete.n))
        self.assertAlmostEqual(sample['t'][0].sum(), 1)
        self.assertArrayShape(sample['t'][1], (1, *box.shape))
        self.assertArrayShape(sample['n']['d'], (1, discrete.n))
        self.assertAlmostEqual(sample['n']['d'].sum(), 1)
        self.assertArrayShape(sample['n']['b'], (1, *box.shape))

        mode = dist.mode(dist.default_priors)
        self.assertEqual(mode.keys(), dist._structure.keys())
        self.assertArrayShape(mode['t'][0], (1, discrete.n))
        self.assertAlmostEqual(mode['t'][0].sum(), 1)
        self.assertArrayShape(mode['t'][1], (1, *box.shape))
        self.assertArrayShape(mode['n']['d'], (1, discrete.n))
        self.assertAlmostEqual(mode['n']['d'].sum(), 1)
        self.assertArrayShape(mode['n']['b'], (1, *box.shape))

    def test_aggregated_quantities(self):
        space = gym.spaces.Dict({
            'foo': gym.spaces.Box(low=0, high=1, shape=(2, 7)),
            'bar': gym.spaces.MultiDiscrete([3, 5]),
        })
        dist = ProbaDist(space)
        params_p = {
            'foo': {
                'mu': jax.random.normal(next(self.rngs), shape=(11, 2, 7)),
                'logvar': jax.random.normal(next(self.rngs), shape=(11, 2, 7)),
            },
            'bar': [
                {'logits': jax.random.normal(next(self.rngs), shape=(11, 3))},
                {'logits': jax.random.normal(next(self.rngs), shape=(11, 5))},
            ],
        }
        params_q = {
            'foo': {
                'mu': jax.random.normal(next(self.rngs), shape=(11, 2, 7)),
                'logvar': jax.random.normal(next(self.rngs), shape=(11, 2, 7)),
            },
            'bar': [
                {'logits': jax.random.normal(next(self.rngs), shape=(11, 3))},
                {'logits': jax.random.normal(next(self.rngs), shape=(11, 5))},
            ],
        }

        sample = dist.sample(params_p, next(self.rngs))
        log_proba = dist.log_proba(params_p, sample)
        self.assertArrayShape(log_proba, (11,))
        self.assertTrue(jnp.all(log_proba < 0))

        entropy = dist.entropy(params_p)
        self.assertArrayShape(entropy, (11,))

        cross_entropy = dist.cross_entropy(params_p, params_q)
        self.assertArrayShape(cross_entropy, (11,))
        self.assertTrue(jnp.all(cross_entropy > entropy))

        kl_divergence = dist.kl_divergence(params_p, params_q)
        self.assertArrayShape(kl_divergence, (11,))
        self.assertTrue(jnp.all(kl_divergence > 0))

        kl_div_from_ce = dist.cross_entropy(params_p, params_q) - dist.entropy(params_p)
        self.assertArrayAlmostEqual(kl_divergence, kl_div_from_ce)

    def test_prepostprocess_variate(self):
        space = gym.spaces.Dict({
            'box': gym.spaces.Box(low=0, high=1, shape=(2, 7)),
            'multidiscrete': gym.spaces.MultiDiscrete([3, 5]),
        })
        dist = ProbaDist(space)
        dist_params = {
            'box': {
                'mu': jax.random.normal(next(self.rngs), shape=(11, 2, 7)),
                'logvar': jax.random.normal(next(self.rngs), shape=(11, 2, 7)),
            },
            'multidiscrete': [
                {'logits': jax.random.normal(next(self.rngs), shape=(11, 3))},
                {'logits': jax.random.normal(next(self.rngs), shape=(11, 5))},
            ],
        }
        X_raw = dist.sample(dist_params, next(self.rngs))
        X_clean = dist.postprocess_variate(next(self.rngs), X_raw, batch_mode=True)
        x_clean = dist.postprocess_variate(next(self.rngs), X_raw, batch_mode=False)
        print(jax.tree_map(jnp.shape, X_raw))
        print(jax.tree_map(jnp.shape, X_clean))
        print(X_clean)
        self.assertArrayShape(X_raw['box'], (11, 2, 7))
        self.assertArrayShape(X_clean['box'], (11, 2, 7))
        self.assertFalse(jnp.all(X_raw['box'] > 0))
        self.assertNotIn(X_raw['box'][0], space['box'])
        self.assertTrue(jnp.all(X_clean['box'] > 0))
        self.assertIn(X_clean['box'][0], space['box'])
        self.assertArrayShape(X_raw['multidiscrete'][0], (11, 3))
        self.assertArrayShape(X_raw['multidiscrete'][1], (11, 5))
        self.assertArrayShape(X_clean['multidiscrete'], (11, 2))
        self.assertNotIn(X_raw['multidiscrete'][0], space['multidiscrete'])
        self.assertIn(X_clean['multidiscrete'][0], space['multidiscrete'])
        self.assertIn(x_clean['multidiscrete'], space['multidiscrete'])
