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

from copy import deepcopy


class ParamMixin:

    @property
    def params(self):
        r"""

        The model parameters, i.e. the weights.

        """
        with self.func_approx._state_lock:
            return {c: self.func_approx.state[c]['params'] for c in self.COMPONENTS}

    @params.setter
    def params(self, new_params):
        if set(self.params) != set(new_params):
            raise ValueError("cannot set new params if keys don't match old params")
        with self.func_approx._state_lock:
            for c in self.COMPONENTS:
                self.func_approx.state[c]['params'] = new_params[c]

    @property
    def function_state(self):
        r"""

        The state of the forward-pass function. See :func:`haiku.transform_with_state` for more
        details.

        """
        with self.func_approx._state_lock:
            return {c: self.func_approx.state[c]['function_state'] for c in self.COMPONENTS}

    @function_state.setter
    def function_state(self, new_function_state):
        if set(self.function_state) != set(new_function_state):
            raise ValueError("cannot set new function_state if keys don't match old function_state")
        with self.func_approx._state_lock:
            for c in self.COMPONENTS:
                self.func_approx.state[c]['function_state'] = new_function_state[c]

    @property
    def hyperparams(self):
        r""" The hyperparameters (if any). """
        return {}

    @hyperparams.setter
    def hyperparams(self, new_hyperparams):
        if set(self.hyperparams) != set(new_hyperparams):
            raise ValueError("cannot set new hyperparams if keys don't match old hyperparams")
        for k, v in new_hyperparams.items():
            setattr(self, k, v)

    def copy(self):
        r"""

        Create a deep copy.

        Returns
        -------
        copy

            A deep copy of the current instance.

        """
        return deepcopy(self)

    def smooth_update(self, other, tau=1.0):
        r"""

        Synchronize the current instance with ``other`` by exponential
        smoothing.

        .. math::

            \theta\leftarrow
                (1 - \tau)\,\theta + \tau\, \theta_\text{new}

        Parameters
        ----------
        other

            A seperate copy of the current object. This object will hold the
            new parameters :math:`\theta_\text{new}`.

        tau : float between 0 and 1, optional

            If we set :math:`\tau=1` we do a hard update. If we pick a smaller
            value, we do a smooth update.

        """
        if not isinstance(other, self.__class__):
            raise TypeError("'self' and 'other' must be of the same type")

        self.params = self.func_approx._smooth_update_func(
            self.params, other.params, tau)
