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

import os
import threading
from zipfile import ZipFile
from tempfile import TemporaryDirectory
from abc import ABC, abstractmethod
from copy import deepcopy

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import optix

from .._base.mixins import SpaceUtilsMixin, RandomStateMixin, LoggerMixin
from ..utils import StrippedEnv, single_to_batch


__all__ = (
    'BaseFuncApprox',
)


class BaseFuncApprox(ABC, SpaceUtilsMixin, RandomStateMixin, LoggerMixin):
    r"""

    This class is the main entry point for constructing a custom function approximator. The
    following methods may be overridden:

    .. hlist::
        :columns: 2

        * :attr:`optimizer`
        * :attr:`body`
        * :attr:`state_action_combiner`
        * :attr:`action_preprocessor`
        * :attr:`action_postprocessor`
        * :attr:`head_v`
        * :attr:`head_pi`
        * :attr:`head_q1`
        * :attr:`head_q2`

    Parameters
    ----------
    env : gym environment

        A gym-style environment.

    strip_env : bool, optional

        Whether to store ``env`` as a static :class:`StrippedEnv <coax.utils.StrippedEnv>`. This is
        particularly useful for avoiding issues with stateful environments when trying to copy
        and/or pickle a function approximator.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    example_observation : state observation, optional

        If left unspecified, we'll set

        .. code:: python

            example_observation = env.observation_space.sample()


    example_action : action, optional

        If left unspecified, we'll set

        .. code:: python

            example_action = env.action_space.sample()


    \*\*optimizer_kwargs

        Keyword arguments to be used in the :attr:`optimizer` method. The optimizer kwargs are
        stored internally as ``self.optimizer_kwargs`` (dict).

    """
    COMPONENTS = (
        'body',
        'head_v',
        'head_q1',
        'head_q2',
        'head_pi',
        'action_preprocessor',
        'action_postprocessor',
        'state_action_combiner',
    )

    def __init__(
            self, env,
            strip_env=True,
            random_seed=None,
            example_observation=None,
            example_action=None,
            **optimizer_kwargs):

        self.strip_env = bool(strip_env)
        self.random_seed = random_seed
        self.env = StrippedEnv(env, self.random_seed) if self.strip_env else env
        self.optimizer_kwargs = optimizer_kwargs or {'learning_rate': 0.001}
        self._state_lock = threading.Lock()  # careful: not pickleable

    @classmethod
    def from_spaces(
            cls, observation_space, action_space,
            reward_range=None, spec=None, metadata=None,
            random_seed=None, **optimizer_kwargs):
        r"""

        Create a new instance from ``observation_space`` and ``action_space``.

        Parameters
        ----------
        observation_space : gym-style space

            The space of state observations :math:`s`.

        action_space : gym-style space

            The space of actions :math:`a`.

        reward_range : pair of floats, optional

            The range of the rewards generated by the environment.

        spec : EnvSpec, optional

            The environment's EnvSpec. See the :mod:`gym.envs.register` module form more details.

        metadata : dict, optional

            The metadata dict of the environment.

        random_seed : int, optional

            Sets the random state to get reproducible results.

        \*\*optimizer_kwargs

            Keyword arguments to be used in the :attr:`optimizer` method. The optimizer kwargs are
            stored internally as ``self.optimizer_kwargs`` (dict).

        """
        env = StrippedEnv.from_spaces(
            observation_space=observation_space,
            action_space=action_space,
            reward_range=reward_range,
            spec=spec,
            metadata=metadata,
            random_seed=random_seed)
        return cls(env, random_seed=random_seed, **optimizer_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()  # shallow copy
        del state['_state_lock']      # exclude non-pickleable attrs
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)          # restore state
        self._state_lock = threading.Lock()  # reinstate non-pickleable attr

    # ----------------------------------------------------------------------------------------------
    # public methods (abstract)
    # ----------------------------------------------------------------------------------------------

    @abstractmethod
    def body(self, S):
        r"""

        .. note::

            This property typically needs to be provided by the end user.


        This property defines the part of the function approximator that is shared across different
        output types (heads).

        The default implementation does a no-op. It is therefore advisable to implement a more
        appropriate custom body.

        Parameters
        ----------
        S : ndarray

            A batch of state observations :math:`s`.

        Returns
        -------

        X_s : ndarray

            A featurize-extracted version of the input batch of state observations :math:`s`.

        """
        pass

    @abstractmethod
    def head_v(self, X_s):
        r"""

        This is probably the most basic head, which implements the head of a simple state value
        function :math:`v(s)`.

        The default implementation does simple linear regression on top of the output of
        :attr:`body`. The final output is a scalar, to be interpreted as :math:`v(s)`.

        Parameters
        ----------
        X_s : ndarray

            Output of :attr:`body`, which can be viewed as a featurize-extracted version of a batch
            of state observations :math:`s`.

        Returns
        -------
        V : ndarray

            A batch of scalar-valued q-values representing :math:`v(s)\in\mathbb{R}`.

        """
        pass

    @abstractmethod
    def head_q1(self, X_sa):
        r"""

        A type-I q-function models the q-function as

        .. math::

            (s, a) \mapsto  q(s, a) \in \mathbb{R}

        The default implementation does simple linear regression on top of the
        output of :attr:`state_action_combiner` (not :attr:`body`).

        Parameters
        ----------
        X_sa : ndarray

            Output of :attr:`state_action_combiner`, which can be viewed as a
            featurize-extracted version of a batch of state-actions pairs
            :math:`(s, a)`.

        Returns
        -------
        Q_sa : ndarray

            A batch of scalar-valued q-values representing
            :math:`q(s,a)\in\mathbb{R}`.

        """
        pass

    @abstractmethod
    def head_q2(self, X_s):
        r"""

        A type-II q-function models the q-function as

        .. math::

            s \mapsto q(s,.) \in \mathbb{R}^n

        where :math:`n` is the number of actions.

        The default implementation does multi-linear regression on top of the output of
        :attr:`body`.

        For the time being, type-II q-functions are only implemented for discrete actions spaces.

        Parameters
        ----------
        X_s : ndarray

            Output of :attr:`body`, which can be viewed as a featurize-extracted version of a batch
            of state observations :math:`s`.

        Returns
        -------
        Q_s : ndarray

            A batch of vectors representing :math:`q(s, .)\in\mathbb{R}^n`.

        """
        pass

    @abstractmethod
    def head_pi(self, X_s):
        r"""

        Depending on the action space, this may return different output shapes.

        If the action space is discrete, the default implementation does multi-linear regression on
        top of the output of :attr:`body`. The final output can then be interpreted as logits
        :math:`z(s)`, which are conditioned on the state observations :math:`s`. These logits can be
        used as parameters of a categorical distribution. The resulting policy is known as a
        *softmax policy*.

        If the action space is a :class:`Box <gym.spaces.Box>`, the default implementation does two
        copies of multi-linear regression on top of the output of :attr:`body`. The output can be
        interpreted as the mean and the log-variance :math:`(\mu(s), \log\sigma^2(s))` of a
        multi-dimensional Gaussian distribution :math:`\mathcal{N}(\mu(s),\sigma^2(s)\,\mathbb{I})`,
        conditioned on the state observations :math:`s`. The resulting policy is therefore a
        *Gaussian policy*.

        For other action spaces, you will have to implement your own policy head.

        Parameters
        ----------
        X_s : ndarray

            Output of :attr:`body`, which can be viewed as a featurize-extracted version of a batch
            of state observations :math:`s`.

        Returns
        -------
        P : pytree with ndarray leaves

            A batch of policy parameters, representing the conditional probability distribution
            :math:`\pi(.|s)`. In other words, the parameters of the probability distribution depend
            on :math:`s`.

            For a softmax policy, ``P = {'logits': array([...])}``. For a Gaussian policy, this
            should return ``P = {'mu': array([...]), 'logvar': array([...])}`` instead.


        """
        pass

    @abstractmethod
    def state_action_combiner(self, X_s, X_a):
        r"""

        The main task of this property is to combine the state observation input :math:`s` with the
        action input :math:`a`.

        The default implementation takes the output of :attr:`body`, which we may think of as a
        featurized version of the state observation, and then combines it with the input action via
        a Kronecker product. Schematically this looks like

        .. math::

            (s, a) \mapsto
                \texttt{body}(s) \otimes \texttt{action_preprocessor}(a)\ \in \mathbb{R}^n

        where :math:`n` is the product of the output dimensionalities of
        :attr:`body` and :attr:`action_preprocessor`.

        Parameters
        ----------
        X_s : ndarray

            Output of :attr:`body`, which can be viewed as a featurize-extracted version of a batch
            of state observations :math:`s`.

        X_a : ndarray

            Output of :attr:`action_preprocessor`, which can be viewed as a featurize-extracted
            version of a batch of actions :math:`a`.

        Returns
        -------
        X_sa : ndarray

            A batch of vectors, representing a feature-extracted version of batch of state-action
            pairs :math:`(s, a)`.

        """
        pass

    @abstractmethod
    def action_preprocessor(self, A):
        r"""

        Preprocess the actions such that they can be combined with the (preprocessed) state
        observations, which happens in :attr:`q1`.

        The default behavior is to do one-hot encoding if the actions space is discrete and to
        flatten the actions if the action space is a :class:`Box`, i.e. continuous.

        Parameters
        ----------
        A : ndarray

            A batch of actions :math:`a`.

        Returns
        -------
        X_a : ndarray

            A batch of vectors, representing a feature-extracted version of batch of actions
            :math:`a`.

        """
        pass

    @abstractmethod
    def action_postprocessor(self, A):
        r"""

        The inverse transformation of :attr:`action_preprocessor`.

        Parameters
        ----------
        X_a : ndarray

            A batch of vectors, representing a feature-extracted version of batch of actions
            :math:`a`.

        Returns
        -------
        A : ndarray

            A batch of actions :math:`a`.

        """
        pass

    @abstractmethod
    def optimizer(self):
        r"""

        The optimizer used to translate gradients to parameter updates. This uses the
        :mod:`jax.optix` module.

        The default implementation is SGD with Nesterov momentum.

        Returns
        -------
        optimizer

            An :mod:`jax.optix` style optimizer.

        """
        pass

    # ----------------------------------------------------------------------------------------------
    # public methods (implemented)
    # ----------------------------------------------------------------------------------------------

    def update_params(self, **grads):
        r"""

        Update the underlying :attr:`state['params'] <state>` from a collection of gradients.

        Parameters
        ----------
        \*\*grads : pytree with ndarray leaves

            Gradients with respect to the model parameters associated with each component. For
            instance, for a state value function :math:`v(s)` this is of the form:

            .. code:: python

                func_approx.update_params(body={...}, head_v={...})

        """
        with self._state_lock:
            # update params for each component
            for c, g in grads.items():
                self.state[c]['params'], self.state[c]['optimizer_state'] = \
                    self._update_func(self.state[c]['params'], g, self.state[c]['optimizer_state'])

    def update_function_state(self, **function_state):
        r"""

        Update the underlying :attr:`state['function_state'] <state>` from a collection of updated
        internal states of the forward-pass functions. See :func:`haiku.transform_with_state` for
        more details.

        Parameters
        ----------
        \*\*function_state : pytree with ndarray leaves

            Internal state of the forward-pass functions associated with each component. For
            instance, for a state value function :math:`v(s)` this is of the form:

            .. code:: python

                func_approx.update_function_state(body={...}, head_v={...})

        """
        with self._state_lock:
            for c, state in function_state.items():
                self.state[c]['function_state'] = state

    def copy(self):
        r"""

        Create a deep copy.

        Returns
        -------
        func_approx : FuncApprox

            A deep copy of the current instance.

        """
        new = deepcopy(self)
        new.env = self.env
        return new

    def smooth_update(self, other, tau=1.0):
        r"""

        Synchronize the current instance with ``other`` by exponential smoothing.

        .. math::

            \theta\leftarrow
                \theta + \tau\, (\theta_\text{new} - \theta)

        Parameters
        ----------
        other

            A seperate copy of the current object. This object will hold the new parameters
            :math:`\theta_\text{new}`.

        tau : float between 0 and 1, optional

            If we set :math:`\tau=1` we do a hard update. If we pick a smaller value, we do a smooth
            update.

        """
        if not isinstance(other, self.__class__):
            raise TypeError("'self' and 'other' must be of the same type")
        with self._state_lock:
            for c in self.state:
                self.state[c]['params'] = self._smooth_update_func(
                    self.state[c]['params'], other.state[c]['params'], tau)

    # ----------------------------------------------------------------------------------------------
    # serialization / deserialization
    # ----------------------------------------------------------------------------------------------

    def save_params(self, filepath, save_full_state=True):
        assert isinstance(filepath, str), 'filepath must be a string'
        if not filepath.endswith('.zip'):
            filepath += '.zip'

        # make sure dir exists
        os.makedirs(os.path.abspath(os.path.dirname(filepath)), exist_ok=True)

        with self._params_lock, TemporaryDirectory() as d, ZipFile(filepath, 'w') as z:
            self._write_leaves_to_zipfile('params', d, z)
            if save_full_state:
                self._write_leaves_to_zipfile('optimizer_state', d, z)
                self._write_leaves_to_zipfile('_random_seed', d, z)
                self._write_leaves_to_zipfile('_random_key', d, z)

        self.logger.info(f"saved params to: {filepath}")

    def restore_params(self, filepath):
        try:
            with self._params_lock, TemporaryDirectory() as d, ZipFile(filepath) as z:
                self._load_leaves_from_zipfile('params', d, z, True)
                self._load_leaves_from_zipfile('optimizer_state', d, z, False)
                self._load_leaves_from_zipfile('_random_seed', d, z, False)
                self._load_leaves_from_zipfile('_random_key', d, z, False)
        except Exception:
            self.logger.error("failed to restore weights")
            raise

        self.logger.info(f"restored params from: {filepath}")

    # -------------------------------------------------------------------------
    # private methods
    # -------------------------------------------------------------------------

    def _write_leaves_to_zipfile(self, name, tempdir, zipfile):
        fp = os.path.join(tempdir, name + '.npz')
        leaves = jax.tree_leaves(getattr(self, name))
        jnp.savez(fp, *leaves)
        zipfile.write(fp, os.path.basename(fp))

    def _load_leaves_from_zipfile(self, name, tempdir, zipfile, required):
        fn = name + '.npz'
        if fn not in zipfile.namelist():
            if required:
                raise IOError(f"required file missing from zipfile: {fn}")
            return
        fp = os.path.join(tempdir, fn)
        zipfile.extract(fn, tempdir)
        npzfile = jnp.load(fp)
        treedef = jax.tree_structure(getattr(self, name))
        leaves = npzfile.values()
        pytree = jax.tree_unflatten(treedef, leaves)
        setattr(self, name, pytree)

    def _init_state(self, example_observation, example_action):
        r"""

        This private method initializes the state of the instance.

        """
        self.random_seed = self.random_seed  # this resets the PRNGKey
        self.apply_funcs = {}
        self.state = {}
        self.optimizer_state = {}

        # get example inputs
        if example_observation is not None:
            s = example_observation
        else:
            s = self.env.observation_space.sample()
        if example_action is not None:
            a = example_action
        else:
            a = self.env.action_space.sample()
        S = single_to_batch(self._preprocess_state(s))
        A = single_to_batch(a)
        opt = self.optimizer()

        # body (a.k.a. state-observation preprocessor)
        func = hk.transform_with_state(self.body)
        params, state = func.init(self.rng, S, is_training=True)
        self.apply_funcs['body'] = func.apply
        self.state['body'] = jax.device_put({
            'params': params,
            'function_state': state,
            'optimizer_state': opt.init(params),
        })
        X_s, _ = func.apply(params, state, self.rng, S, is_training=True)

        # action preprocessor (must be stateless)
        func = hk.transform(self.action_preprocessor, apply_rng=True)
        params = func.init(self.rng, A)
        self.apply_funcs['action_preprocessor'] = func.apply
        self.state['action_preprocessor'] = jax.device_put({
            'params': params,
            'function_state': None,
            'optimizer_state': opt.init(params),
        })
        X_a = func.apply(params, self.rng, A)

        # action postprocessor (must be stateless)
        func = hk.transform(self.action_postprocessor, apply_rng=True)
        params = func.init(self.rng, X_a)
        self.apply_funcs['action_postprocessor'] = func.apply
        self.state['action_postprocessor'] = jax.device_put({
            'params': params,
            'function_state': None,
            'optimizer_state': jax.device_put(opt.init(params)),
        })

        # state-action combiner
        func = hk.transform_with_state(self.state_action_combiner)
        params, state = func.init(self.rng, X_s, X_a, is_training=True)
        self.apply_funcs['state_action_combiner'] = func.apply
        self.state['state_action_combiner'] = jax.device_put({
            'params': params,
            'function_state': state,
            'optimizer_state': opt.init(params),
        })
        X_sa, _ = func.apply(params, state, self.rng, X_s, X_a, is_training=True)

        # state value head
        func = hk.transform_with_state(self.head_v)
        params, state = func.init(self.rng, X_s, is_training=True)
        self.apply_funcs['head_v'] = func.apply
        self.state['head_v'] = jax.device_put({
            'params': params,
            'function_state': state,
            'optimizer_state': opt.init(params),
        })

        # state-action value head (type-I)
        func = hk.transform_with_state(self.head_q1)
        params, state = func.init(
            self.rng, X_sa, is_training=True)
        self.apply_funcs['head_q1'] = func.apply
        self.state['head_q1'] = jax.device_put({
            'params': params,
            'function_state': state,
            'optimizer_state': opt.init(params),
        })

        # state-action value head (type-II)
        if self.action_space_is_discrete:
            func = hk.transform_with_state(self.head_q2)
            params, state = func.init(
                self.rng, X_s, is_training=True)
            self.apply_funcs['head_q2'] = func.apply
            self.state['head_q2'] = jax.device_put({
                'params': params,
                'function_state': state,
                'optimizer_state': opt.init(params),
            })
        else:
            self.apply_funcs['head_q2'] = None
            self.state['head_q2'] = None

        # policy head
        func = hk.transform_with_state(self.head_pi)
        params, state = func.init(self.rng, X_s, is_training=True)
        self.apply_funcs['head_pi'] = func.apply
        self.state['head_pi'] = jax.device_put({
            'params': params,
            'function_state': state,
            'optimizer_state': opt.init(params),
        })

        # check if action processors are stateless
        if jax.tree_leaves(self.state['action_preprocessor']['params']):
            raise NotImplementedError(
                "action_preprocessor cannot be stateful. The reason is that we haven't yet "
                "implemented any automated learning for its inverse, i.e. action_postprocessor. "
                "Please submit a feature request if you'd like this functionality implemented.")
        if jax.tree_leaves(self.state['action_postprocessor']['params']):
            raise NotImplementedError(
                "action_postprocessor cannot be stateful. The reason is that we haven't yet "
                "implemented any automated learning for it. Please submit a feature request if "
                "you'd like this functionality implemented.")

        # construct jitted param update function
        def update_func(params, grads, opt_state):
            updates, new_opt_state = opt.update(grads, opt_state)
            new_params = optix.apply_updates(params, updates)
            return new_params, new_opt_state

        def smooth_update_func(old, new, tau):
            return jax.tree_multimap(lambda a, b: (1 - tau) * a + tau * b, old, new)

        self._update_func = jax.jit(update_func)
        self._smooth_update_func = jax.jit(smooth_update_func)
