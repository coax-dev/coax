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
import pickle

import numpy as onp
import jax


class SerializationMixin:
    def _write_leaves_to_zipfile(self, name, tempdir, zipfile):
        fp = os.path.join(tempdir, name + '.npz')
        leaves = jax.tree_leaves(getattr(self, name))
        onp.savez(fp, *leaves)
        zipfile.write(fp, os.path.basename(fp))

    def _load_leaves_from_zipfile(self, name, tempdir, zipfile, required):
        fn = name + '.npz'
        if fn not in zipfile.namelist():
            if required:
                raise IOError(f"required file missing from zipfile: {fn}")
            return False
        fp = os.path.join(tempdir, fn)
        zipfile.extract(fn, tempdir)
        npzfile = onp.load(fp)
        treedef = jax.tree_structure(getattr(self, name))
        leaves = npzfile.values()
        pytree = jax.tree_unflatten(treedef, leaves)
        setattr(self, name, pytree)
        return True

    def _write_pickle_to_zipfile(self, name, tempdir, zipfile):
        zipfile.writestr(name + '.pkl', pickle.dumps(getattr(self, name)))

    def _load_pickle_from_zipfile(self, name, tempdir, zipfile, required):
        fn = name + '.pkl'
        if fn not in zipfile.namelist():
            if required:
                raise IOError(f"required file missing from zipfile: {fn}")
            return False
        fp = os.path.join(tempdir, fn)
        zipfile.extract(fn, tempdir)
        with open(fp, 'rb') as f:
            setattr(self, name, pickle.load(f))
        return True
