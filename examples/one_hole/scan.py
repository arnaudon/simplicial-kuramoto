"""Scan frustration parameter."""
import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

if __name__ == "__main__":
    t_max = 1000
    n_t = 500
    n_workers = 10
    repeats = 10
    n_alpha2 = 200

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
    # with edge 6 flipped
    Gsc.flip_edge_orientation(6)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename="complex_0.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )

    # with edge 6 and 5 flipped
    Gsc.flip_edge_orientation(5)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename="complex_1.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )

    # with edge 6, 5 and 4 flipped
    Gsc.flip_edge_orientation(4)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename="complex_2.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )
