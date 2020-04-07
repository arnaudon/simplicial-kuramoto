#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

setup(
    name="simplicial-kuramoto",
    author="Alexis, Giovani and Paul",
    author_email="alexis.arnaudon@epfl.ch",
    version='0.0.1',
    description="",
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=0.13.3",
        "h5py>=2.9.0",
        "morphio>=2.3.4",
        "matplotlib>=2.2.0",
        "pandas>=0.24.0",
    ],
    packages=find_packages(),
)
