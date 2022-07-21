"""

To update the requirements files, first do a pip freeze in google colab and then past the output in
requirements.colab.txt

After that, just run this script from the project root (where this script is located).

"""
import re
from sys import stderr
from packaging.version import parse as _parse

import pandas as pd


kwargs = {
    'sep': r'(?:==|>=|\s@\s)',
    'names': ['name', 'version'],
    'index_col': 'name',
    'comment': '#',
    'engine': 'python',
}


def parse(row):
    version_str = '' if row.version is None else str(row.version)
    if version_str.startswith('http'):
        m = re.search(row.name + r'\-([\d\.]+)\+', row.version)
        version_str = '' if m is None else m.group(1)
    return _parse(version_str)


def upgrade_requirements(filename):
    reqs_colab = pd.read_csv('requirements.colab.txt', **kwargs)
    reqs_colab['version'] = reqs_colab.apply(parse, axis=1)
    reqs = pd.read_csv(filename, **kwargs)
    reqs['version'] = reqs.apply(parse, axis=1)
    overlap = pd.merge(reqs, reqs_colab, left_index=True, right_index=True, suffixes=('', '_colab'))
    need_updating = overlap[overlap['version'] < overlap['version_colab']]

    with open(filename) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        for name, row in need_updating.iterrows():
            if line.startswith(name):
                new_version = row['version_colab'].base_version
                line_new = line.replace(str(row['version']), new_version)
                lines[i] = line_new
                print(f"updating {filename}: {line.rstrip()} --> {line_new.rstrip()}", file=stderr)

    with open(filename, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    upgrade_requirements('requirements.txt')
    upgrade_requirements('requirements.dev.txt')
    upgrade_requirements('requirements.doc.txt')
