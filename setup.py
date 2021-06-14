#!/usr/bin/env python

import imp
import sys

from setuptools import find_packages, setup

setup(
    name="simplicial-kuramoto",
    author="Alexis, Giovani and Paul",
    author_email="alexis.arnaudon@epfl.ch",
    version="0.0.1",
    description="",
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=0.13.3",
        "matplotlib>=2.2.0",
        "networkx>=2.5",
    ],
    packages=find_packages(),
)
