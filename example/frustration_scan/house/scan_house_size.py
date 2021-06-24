import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters


def my_house(size):

    if size < 3:
        print("I can't build a house with this ...")
        return

    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)

    for i in range(size - 2):
        G.add_edge(2 + i, 3 + i, weight=1)

    G.add_edge(i + 3, 1, weight=1)
    if size == 3:
        Gsc = SimplicialComplex(graph=G, faces=[[0,1,2]])
    else:
        Gsc = SimplicialComplex(graph=G)

    Gsc.flip_edge_orientation([0, size + 1])
    return Gsc


if __name__ == "__main__":
    sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    alpha1 = np.linspace(0, 2.2, 100)
    alpha2 = np.linspace(0, np.pi, 100)
    n_repeats = 1
    n_workers = 80

    t_max = 1000
    n_t = 100

    for size in sizes:
        Gsc = my_house(size)

        scan_frustration_parameters(
            Gsc,
            filename=f"house_{size}.pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=n_workers,
            t_max=t_max,
            n_t=n_t,
        )
