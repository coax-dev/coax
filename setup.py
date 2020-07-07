#!/usr/bin/env python3
# flake8: noqa
import re
import os
import setuptools


pwd = os.path.dirname(__file__)

def get_install_requires(requirements_txt):
    install_requires = []
    with open(os.path.join(pwd, requirements_txt)) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                install_requires.append(line)
    return install_requires


def get_version():
    with open(os.path.join(pwd, 'coax', '__init__.py')) as f:
        version_match = re.search(r'__version__ \= \'(\d+\.\d+\.\d+)\'', f.read())
    assert version_match is not None, "can't parse __version__ from __init__.py"
    version = version_match.group(1)
    assert len(version.split('.')) == 3, "bad version spec"
    return version  # e.g. "0.1.2"


def get_dev_status():
    dev_status = {
        '0.1': 'Development Status :: 1 - Planning',          # v0.1 - skeleton
        '0.2': 'Development Status :: 2 - Pre-Alpha',         # v0.2 - some basic functionality
        '0.3': 'Development Status :: 3 - Alpha',             # v0.3 - most functionality
        '0.4': 'Development Status :: 4 - Beta',              # v0.4 - most functionality + doc
        '1.0': 'Development Status :: 5 - Production/Stable', # v1.0 - most functionality + doc + test  # noqa
        '2.0': 'Development Status :: 6 - Mature',            # v2.0 - new functionality?
    }
    version = get_version()
    majorminor = version.rsplit('.', 1)[0]  # e.g. "0.1"
    return dev_status[majorminor]



long_description = """
coax: Plug-n-play reinforcement learning in python with OpenAI Gym and JAX.

For full documentation, go to:

    https://coax.readthedocs.io

You can find the source code at:

    https://github.com/microsoft/coax

"""


# main setup kw args
setup_kwargs = {
    'name': 'coax',
    'version': get_version(),
    'description': "Plug-n-play reinforcement learning with OpenAI Gym and JAX",
    'long_description': long_description,
    'author': 'Kristian Holsheimer',
    'author_email': 'kristian.holsheimer@gmail.com',
    'url': 'https://coax.readthedocs.io',
    'license': 'MIT',
    'install_requires': get_install_requires('requirements.txt'),
    'extras_require': {
        'dev':  get_install_requires('requirements.dev.txt'),
    },
    'classifiers': [
        get_dev_status(),
        'Environment :: Other Environment',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    'zip_safe': True,
    'packages': setuptools.find_packages(exclude=['test_*.py', '*_test.py']),
}


if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)
