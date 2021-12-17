#!/usr/bin/env python3
import re
import os
import setuptools
from collections import namedtuple

PROJECTDIR = os.path.dirname(__file__)
RE_VERSION = re.compile(
    r'^__version__ \= \'(?P<version>(?P<majorminor>\d+\.\d+)\.\d+(?:\w+\d+)?)\'$', re.MULTILINE)
DEV_STATUS = {
    '0.1': 'Development Status :: 1 - Planning',           # v0.1 - skeleton
    '0.2': 'Development Status :: 2 - Pre-Alpha',          # v0.2 - some basic functionality
    '0.3': 'Development Status :: 3 - Alpha',              # v0.3 - most functionality
    '0.4': 'Development Status :: 4 - Beta',               # v0.4 - most functionality + doc
    '1.0': 'Development Status :: 5 - Production/Stable',  # v1.0 - most functionality + doc + test
    '2.0': 'Development Status :: 6 - Mature',             # v2.0 - new functionality?
}

VersionSpec = namedtuple('VersionSpec', 'version majorminor')


def get_install_requires(requirements_txt):
    install_requires = []
    with open(os.path.join(PROJECTDIR, requirements_txt)) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                install_requires.append(line)
    return install_requires


def get_version_spec():
    with open(os.path.join(PROJECTDIR, 'coax', '__init__.py')) as f:
        version_match = re.search(RE_VERSION, f.read())
    assert version_match is not None, "can't parse __version__ from __init__.py"
    version_spec = VersionSpec(**version_match.groupdict())
    return version_spec


def get_long_description():
    with open(os.path.join(PROJECTDIR, 'README.rst')) as f:
        return f.read()


version_spec = get_version_spec()


# main setup kw args
setup_kwargs = {
    'name': 'coax',
    'version': version_spec.version,
    'description': "Plug-n-play reinforcement learning with OpenAI Gym and JAX",
    'long_description': get_long_description(),
    'author': 'Kristian Holsheimer',
    'author_email': 'kristian.holsheimer@gmail.com',
    'url': 'https://coax.readthedocs.io',
    'license': 'MIT',
    'python_requires': '~=3.6',
    'install_requires': get_install_requires('requirements.txt'),
    'extras_require': {
        'dev': get_install_requires('requirements.dev.txt'),
        'doc': get_install_requires('requirements.doc.txt'),
        'ray': ['ray>1.9.0'],
    },
    'classifiers': [
        DEV_STATUS[version_spec.majorminor],
        'Environment :: GPU :: NVIDIA CUDA',
        'Framework :: Flake8',
        'Framework :: IPython',
        'Framework :: Jupyter',
        'Framework :: Pytest',
        'Framework :: Sphinx',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    'zip_safe': True,
    'packages': setuptools.find_packages(),
}


if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)
