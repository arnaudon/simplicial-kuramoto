#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="simplicial-kuramoto",
    author="Alexis Arnaudon",
    author_email="alexis.arnaudon@epfl.ch",
    version="0.0.1",
    description="",
    install_requires=[
        "numpy>=1.15.0",
        "scipy>=0.13.3",
        "matplotlib>=2.2.0",
        "networkx>=2.5",
        "pandas>=1.0.2",
        "tqdm",
        "nolds",
    ],
    packages=find_packages(),
)
