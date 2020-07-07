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

from abc import ABC, abstractmethod

import jax

from ..mixins import SpaceUtilsMixin


__all__ = (
    'BaseFunc',
)


class BaseFunc(ABC, SpaceUtilsMixin):

    def __init__(self, func_approx):
        self.func_approx = func_approx
        self._init_action_processors()

    @property
    def env(self):
        r""" The main gym-style environment. """
        return self.func_approx.env

    @property
    def rng(self):
        return self.func_approx.rng

    @property
    def random_seed(self):
        return self.func_approx.random_seed

    @random_seed.setter
    def random_seed(self, new_random_seed):
        self.func_approx.random_seed = new_random_seed

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_eval(self, *args, **kwargs):
        pass

    def _init_action_processors(self):
        def action_preprocessor_func(params, rng, A):
            func = self.func_approx.apply_funcs['action_preprocessor']
            return func(params['action_preprocessor'], rng, A)

        def action_postprocessor_func(params, rng, X_a):
            func = self.func_approx.apply_funcs['action_postprocessor']
            return func(params['action_postprocessor'], rng, X_a)

        self._action_preprocessor_func = jax.jit(action_preprocessor_func)
        self._action_postprocessor_func = jax.jit(action_postprocessor_func)

    @property
    def action_preprocessor_func(self):
        r"""

        JIT-compiled function responsible for preprocessing a batch of actions
        in such a way that they can be fed into :attr:`log_proba` method of the
        underlying :attr:`proba_dist`. See also
        :attr:`coax.FuncApprox.action_preprocessor`.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying
            q-function.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        A : actions

            A batch of actions.

        Returns
        -------
        X_a : transformed actions

            A batch of actions that are transformed in such a way that can they
            be fed into :attr:`log_proba` method of the underlying
            :attr:`proba_dist`.

            Note that these actions cannot be fed directly into a gym-style
            environment. For example, if the action space is discrete, these
            transformed actions are (approximately) one-hot encoded. This means
            that we need to apply an :func:`argmax <coax.utils.argmax>` before
            we can feed the actions into a gym-style environment.

        """
        return self._action_preprocessor_func

    @property
    def action_postprocessor_func(self):
        r"""

        JIT-compiled function responsible for doing the inverse
        :attr:`action_preprocessor_func`. See also
        :attr:`coax.FuncApprox.action_postprocessor`.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying
            q-function.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        X_a : transformed actions

            A batch of actions that are transformed in such a way that can be
            fed into :attr:`log_proba` method of the underlying
            :attr:`proba_dist`.

        Returns
        -------

        A : actions

            A batch of actions in their original, gym-environment compatible
            form.

        """
        return self._action_postprocessor_func
