import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

if __name__ == "__main__":

    G = nx.Graph()
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(2, 0, weight=1, edge_com=0)

    Gsc = SimplicialComplex(graph=G)
    Gsc.flip_edge_orientation([0, 1])

    alpha1 = np.linspace(0, 2.5, 10)
    alpha2 = np.linspace(0, np.pi / 2.0, 20)
    alpha3 = np.array([0.])

    n_repeats = 1
    t_max = 500
    n_t = 1000
    n_workers = 80

    scan_frustration_parameters(
        Gsc,
        filename="triangle.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        repeats=n_repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
    )
