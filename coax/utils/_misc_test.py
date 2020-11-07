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
import tempfile

from ._misc import dump, dumps, load, loads


def test_dump_load():
    with tempfile.TemporaryDirectory() as d:
        a = [13]
        b = {'a': a}

        # references preserved
        dump((a, b), os.path.join(d, 'ab.pkl.lz4'))
        a_new, b_new = load(os.path.join(d, 'ab.pkl.lz4'))
        b_new['a'].append(7)
        assert b_new['a'] == [13, 7]
        assert a_new == [13, 7]

        # references not preserved
        dump(a, os.path.join(d, 'a.pkl.lz4'))
        dump(b, os.path.join(d, 'b.pkl.lz4'))
        a_new = load(os.path.join(d, 'a.pkl.lz4'))
        b_new = load(os.path.join(d, 'b.pkl.lz4'))
        b_new['a'].append(7)
        assert b_new['a'] == [13, 7]
        assert a_new == [13]


def test_dumps_loads():
    a = [13]
    b = {'a': a}

    # references preserved
    s = dumps((a, b))
    a_new, b_new = loads(s)
    b_new['a'].append(7)
    assert b_new['a'] == [13, 7]
    assert a_new == [13, 7]

    # references not preserved
    s_a = dumps(a)
    s_b = dumps(b)
    a_new = loads(s_a)
    b_new = loads(s_b)
    b_new['a'].append(7)
    assert b_new['a'] == [13, 7]
    assert a_new == [13]
