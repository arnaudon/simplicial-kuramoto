import numpy as np
import sys
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it
import pickle
from tqdm import tqdm
import pandas as pd
import copy
from numpy import linalg as LA

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters, proj_subspace
from simplicial_kuramoto.graph_generator import delaunay_with_holes
from simplicial_kuramoto.frustration_scan import (
    get_subspaces,
    compute_order_parameter,
    plot_order_1d,
    _get_projections_1d,
)
from simplicial_kuramoto.plotting import draw_simplicial_complex

if __name__ == "__main__":
    t_max = 1000
    n_t = 500
    n_workers = 10
    repeats = 10
    n_alpha2 = 200

    # The candidate
    G = nx.Graph()

    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(0, 3, weight=1, edge_com=0)
    G.add_edge(1, 3, weight=1, edge_com=0)
    G.add_edge(1, 4, weight=1, edge_com=0)
    G.add_edge(2, 4, weight=1, edge_com=0)

    G.add_edge(3, 5, weight=1, edge_com=0)
    G.add_edge(3, 6, weight=1, edge_com=0)
    G.add_edge(4, 6, weight=1, edge_com=0)
    G.add_edge(4, 7, weight=1, edge_com=0)
    G.add_edge(5, 6, weight=1, edge_com=0)
    G.add_edge(6, 7, weight=1, edge_com=0)

    G.add_edge(1, 6, weight=1, edge_com=0)

    # pos = nx.spring_layout(G,)
    pos_ = {}
    pos_[0] = np.array([0, 0])
    pos_[1] = np.array([1, 0])
    pos_[2] = np.array([2, 0])
    pos_[3] = np.array([0, 1])
    pos_[4] = np.array([2, 1])
    pos_[5] = np.array([0, 2])
    pos_[6] = np.array([1, 2])
    pos_[7] = np.array([2, 2])

    for n in G.nodes:
        G.nodes[n]["pos"] = pos_[n]

    Gsc = SimplicialComplex(graph=G, no_faces=False)

    del Gsc.faces[3]

    ## original orientation
    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename=f"Fig_2_example_1_1.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )

    ## with edge 6 flipped
    Gsc.flip_edge_orientation(6)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename=f"Fig_2_example_1_2.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )

    ## with edge 6 and 5 flipped
    Gsc.flip_edge_orientation(5)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename=f"Fig_2_example_1_3.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )

    ## with edge 6, 5 and 4 flipped
    Gsc.flip_edge_orientation(4)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename=f"Fig_2_example_1_4.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )

    ## with edge 6, 5 and 9 flipped
    Gsc.flip_edge_orientation(4)
    Gsc.flip_edge_orientation(9)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename=f"Fig_2_example_1_5.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )
