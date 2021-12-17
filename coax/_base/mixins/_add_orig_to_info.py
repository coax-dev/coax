from typing import Mapping


class AddOrigToInfoDictMixin:
    def _add_s_orig_to_info_dict(self, info):
        if not isinstance(info, Mapping):
            assert info is None, "unexpected type for 'info' Mapping"
            info = {}

        if 's_orig' in info:
            info['s_orig'].append(self._s_orig)
        else:
            info['s_orig'] = [self._s_orig]

        if 's_next_orig' in info:
            info['s_next_orig'].append(self._s_next_orig)
        else:
            info['s_next_orig'] = [self._s_next_orig]

        self._s_orig = self._s_next_orig

    def _add_a_orig_to_info_dict(self, info):
        if not isinstance(info, Mapping):
            assert info is None, "unexpected type for 'info' Mapping"
            info = {}

        if 'a_orig' in info:
            info['a_orig'].append(self._a_orig)
        else:
            info['a_orig'] = [self._a_orig]
