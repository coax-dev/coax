from abc import ABC, abstractmethod

import gym
import jax

from ..utils import batch_to_single


class BaseProbaDist(ABC):
    r"""

    Abstract base class for probability distributions. Check out
    :class:`coax.proba_dists.CategoricalDist` for a specific example.

    """
    __slots__ = (
        '_space',
        '_sample_func',
        '_mean_func',
        '_mode_func',
        '_log_proba_func',
        '_entropy_func',
        '_cross_entropy_func',
        '_kl_divergence_func',
        '_affine_transform_func',
        '_default_priors_func',
    )

    def __init__(self, space):
        if not isinstance(space, gym.Space):
            raise TypeError("space must be derived from gym.Space")
        self._space = space

    @property
    def space(self):
        r""" The gym-style space that specifies the domain of the distribution. """
        return self._space

    @property
    def hyperparams(self):
        r""" The distribution hyperparameters. """
        return {}

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        rng : PRNGKey

            A key for seeding the pseudo-random number generator.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._sample_func

    @property
    def mean(self):
        r"""

        JIT-compiled functions that generates differentiable means of the distribution.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._mean_func

    @property
    def mode(self):
        r"""

        JIT-compiled functions that generates differentiable modes of the distribution.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._mode_func

    @property
    def log_proba(self):
        r"""

        JIT-compiled function that evaluates log-probabilities.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        X : ndarray

            A batch of variates, e.g. a batch of actions :math:`a` collected from experience.

        Returns
        -------
        logP : ndarray of floats

            A batch of log-probabilities associated with the provided variates.

        """
        return self._log_proba_func

    @property
    def entropy(self):
        r"""

        JIT-compiled function that computes the entropy of the distribution.

        .. math::

            H\ =\ -\mathbb{E}_p \log p


        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        H : ndarray of floats

            A batch of entropy values.

        """
        return self._entropy_func

    @property
    def cross_entropy(self):
        r"""

        JIT-compiled function that computes the cross-entropy of a distribution :math:`q` relative
        to another categorical distribution :math:`p`:

        .. math::

            \text{CE}[p,q]\ =\ -\mathbb{E}_p \log q

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._cross_entropy_func

    @property
    def kl_divergence(self):
        r"""

        JIT-compiled function that computes the Kullback-Leibler divergence of a categorical
        distribution :math:`q` relative to another distribution :math:`p`:

        .. math::

            \text{KL}[p,q]\ = -\mathbb{E}_p \left(\log q -\log p\right)

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._kl_divergence_func

    @property
    def affine_transform(self):
        r"""

        Transform the distribution :math:`\mathcal{D}\to\mathcal{D}'` in such a way that its
        associated variables :math:`X\sim\mathcal{D}` and :math:`X'\sim\mathcal{D}'` are related
        via an affine transformation:

        .. math::

            X' = X\times\text{scale} + \text{shift}

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the original distribution :math:`\mathcal{D}`.

        scale : float or ndarray

            The multiplicative factor of the affine transformation.

        shift : float or ndarray

            The additive shift of the affine transformation.

        value_transform : ValueTransform, optional

            The transform to apply to the values before the affine transform, i.e.

            .. math::

                X' = f\bigl(f^{-1}(X)\times\text{scale} + \text{shift}\bigr)

        Returns
        -------
        dist_params : pytree with ndarray leaves

            The distribution parameters of the transformed distribution :math:`\mathcal{D}'`.

        """
        return self._affine_transform_func

    @property
    def dist_params_structure(self):
        r""" The tree structure of the distribution parameters. """
        return jax.tree_structure(self.default_priors)

    @property
    @abstractmethod
    def default_priors(self):
        r""" The default distribution parameters. """
        pass

    def postprocess_variate(self, rng, X, index=0, batch_mode=False):
        r"""

        The post-processor specific to variates drawn from this ditribution.

        This method provides the interface between differentiable, batched variates, i.e. outputs
        of :func:`sample` and :func:`mode` and the provided gym space.

        Parameters
        ----------
        rng : PRNGKey

            A key for seeding the pseudo-random number generator.

        X : raw variates

            A batch of **raw** clean variates, i.e. same format as the outputs of :func:`sample`
            and :func:`mode`.

        index : int, optional

            The index to pick out from the batch. Note that this only applies if
            :code:`batch_mode=False`.

        batch_mode : bool, optional

            Whether to return a batch or a single instance.

        Returns
        -------
        x or X : clean variate

            A single clean variate or a batch thereof (if ``batch_mode=True``). A variate is called
            **clean** if it is an instance of the gym-style :attr:`space`, i.e. it satisfies
            :code:`x in self.space`.

        """
        return X if batch_mode else batch_to_single(X)

    def preprocess_variate(self, rng, X):
        r"""

        The pre-processor to ensure that an instance of the :attr:`space` is processed into the same
        structure as variates drawn from this ditribution, i.e. outputs of :func:`sample` and
        :func:`mode`.

        Parameters
        ----------
        rng : PRNGKey

            A key for seeding the pseudo-random number generator.

        X : clean variates

            A batch of clean variates, i.e. instances of the gym-style :attr:`space`.

        Returns
        -------
        X : raw variates

            A batch of **raw** clean variates, i.e. same format as the outputs of :func:`sample`
            and :func:`mode`.

        """
        return X
