[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7988477.svg)](https://doi.org/10.5281/zenodo.7988477)


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

It also contains the models introduced in
```
A unified framework for Simplicial Kuramoto models
by: Marco Nurisso, Alexis Arnaudon, Maxime Lucas, Robert L. Peach, Paul Expert, Francesco Vaccarino, Giovanni Petri
at: https://arxiv.org/abs/2305.17977
```

and the connectome example in `example/connectome_example`, based on  
```
M. Pope, M. Fukushima, R. F. Betzel, and O. Sporns,
Modular origins of high-amplitude cofluctuations in fine-scale functional connectivity dynamics, Proc. Natl. Acad.
Sci. U.S.A. 118, e2109380118 (2021).
```

# Installation

To install, it is on pypi.org, hence just do:
```
pip install simplicial-kuramoto
```

# Usage

The documentation is available here: https://arnaudon.github.io/simplicial-kuramoto/, and the structure of the code is as follow:

- The module `simplicial_complex.py` extends networkx graph to include faces and computed cached values of various graph theoretical operators such as boundary operators, or Hodge Laplacians.
- The module `meausures.py` contains some measurements functions of the dynamics such as order parameters.
- The module `graph_generator.py` contains a few functions to build simplicial complexes.
- The module `integrators.py` solve the Kuramoto model on a given simplicial complex.
- The module `plotting.py` has some plotting functions of the simplicial complex and the Kuramoto solution.
- The module `frustration_scan.py` contains analysis tools to study frustration of simplicial Kuramoto.

In the folder `examples` are present scripts to generate the figures of the first paper as well as run some of the models of the second paper.
