import os
import tempfile

from ..utils import jit
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


def test_dumps_loads_jitted_function():
    @jit
    def f(x):
        return 13 * x

    # references preserved
    s = dumps(f)
    f_new = loads(s)
    assert f_new(11) == f(11) == 143
