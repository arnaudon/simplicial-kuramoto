# Simplicial Kuramoto

This repository contains the code written to produce the figures of
```
Connecting Hodge and Sakaguchi-Kuramoto: a mathematical framework for coupled
oscillators on simplicial complexes
by: Alexis Arnaudon, Robert L. Peach, Giovanni Petri and Paul Expert
at: https://arxiv.org/pdf/2111.11073.pdf
```

It numerically solves the Kuramoto model on simplicial complexes of order 1 (a graph) and 2 (a graph with faces), with node/edge/face weights and frustration.
A small suite of analysis tool is available to compute the Hodge decomposition of the solution, the simplicial order parameter or the largest Lyapunov exponent.

# Installation

To install, clone the repository, and run in the main folder
```
pip install simplicial-kuramoto
```

# Usage

- The module `simplicial_complex.py` extends networkx graph to include faces and computed cached values of various graph theoretical operators such as boundary operators, or Hodge Laplacians.
- The module `graph_generator.py` contains a few functions to build simplicial complexes.
- The module `integrators.py` solve the Kuramoto model on a given simplicial complex.
- The module `plotting.py` has some plotting functions of the simplicial complex and the Kuramoto solution.
- The module `frustration_scan.py` contains analysis tools to study frustration of simplicial Kuramoto.

In the folder `examples` are present scripts to generate the figures of the paper. 
