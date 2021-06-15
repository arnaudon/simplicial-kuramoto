import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters


def make_loop(size):
    G = nx.Graph()
    for i in range(size - 1):
        G.add_edge(i, i + 1, weight=1)
    G.add_edge(size - 1, 0, weight=1)
    return SimplicialComplex(graph=G, no_faces=True)


if __name__ == "__main__":
    sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    alpha1 = np.linspace(0, 2.5, 50)
    alpha2 = np.linspace(0, np.pi, 20)
    n_repeats = 1
    n_workers = 80

    t_max = 100
    n_t = 100
    for size in sizes:
        Gsc = make_loop(size)
        scan_frustration_parameters(
            Gsc,
            filename=f"loop_{size}.pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=n_workers,
            t_max=t_max,
            n_t=n_t,
        )
