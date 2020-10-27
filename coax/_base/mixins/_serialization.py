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

import lz4.frame
import cloudpickle as pickle


class SerializationMixin:

    @classmethod
    def load(cls, filepath):
        r"""

        Load instance from a file.

        Parameters
        ----------
        filepath : str

            The filepath of the stored instance.

        """
        with lz4.frame.open(filepath, 'rb') as f:
            obj = pickle.loads(f.read())
        if not isinstance(obj, cls):
            raise TypeError(f"loaded obj must be an instance of {cls.__name__}, got: {type(obj)}")
        return obj

    def save(self, filepath):
        r"""

        Save instance to a file.

        Parameters
        ----------
        filepath : str

            The filepath to store the instance.

        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with lz4.frame.open(filepath, 'wb') as f:
            f.write(pickle.dumps(self))
