#!/usr/bin/env python

import os
import re

import setuptools

HERE = os.path.dirname(__file__)


def readme_contents() -> str:
    with open(os.path.join(HERE, 'README.md'), 'r') as fd:
        src = fd.read()
    return src


def get_version() -> str:
    version_file = open(os.path.join('cmdstanpy', '_version.py'))
    version_contents = version_file.read()
    return re.search("__version__ = '(.*?)'", version_contents).group(1)


_classifiers = """
Programming Language :: Python :: 3
License :: OSI Approved :: BSD License
Operating System :: OS Independent
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Natural Language :: English
Programming Language :: Python
Topic :: Scientific/Engineering :: Information Analysis
"""

setuptools.setup(
    name='cmdstanpy',
    version=get_version(),
    description='Python interface to CmdStan',
    long_description=readme_contents(),
    long_description_content_type="text/markdown",
    author='Stan Dev Team',
    url='https://github.com/stan-dev/cmdstanpy',
    license_files=['LICENSE.md'],
    packages=['cmdstanpy', 'cmdstanpy.stanfit', 'cmdstanpy.utils'],
    package_data={
        'cmdstanpy': ['py.typed'],
    },
    entry_points={
        'console_scripts': [
            'install_cmdstan=cmdstanpy.install_cmdstan:__main__',
            'install_cxx_toolchain=cmdstanpy.install_cxx_toolchain:__main__',
        ]
    },
    install_requires=[
        'pandas',
        'numpy>=1.21',
        'tqdm',
    ],
    python_requires='>=3.7',
    extras_require={
        'all': [
            'xarray',
        ],
        'docs': [
            'sphinx',
            'sphinx-gallery',
            'sphinx_rtd_theme',
            'numpydoc',
            'matplotlib',
        ],
        'tests': [
            'flake8',
            'pylint',
            'pytest',
            'pytest-cov',
            'pytest-order',
            'mypy',
            'tqdm',
            'xarray',
        ],
    },
    classifiers=_classifiers.strip().split('\n'),
)
