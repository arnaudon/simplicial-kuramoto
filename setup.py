#!/usr/bin/env python3

from setuptools import find_packages, setup

test_require = [
    "pyyaml",
    "dictdiffer",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "diff-pdf-visually",
    "ipython!=8.7.0",  # see https://github.com/spatialaudio/nbsphinx/issues/687
]
setup(
    name="simplicial-kuramoto",
    author="Alexis Arnaudon",
    author_email="alexis.arnaudon@epfl.ch",
    version="0.0.2",
    description="",
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=0.13.3",
        "matplotlib>=2.2.0",
        "networkx>=2.5",
        "pandas>=1.0.2",
        "tqdm",
        "nolds",
        "xgi",
    ],
    extras_require={
        "all": test_require,
        "test": test_require,
    },
    packages=find_packages(),
)
