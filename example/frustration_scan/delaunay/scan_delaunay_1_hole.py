import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import delaunay_with_holes


if __name__ == "__main__":

    np.random.seed(42)
    centres = [[0.5, 0.5]]
    radii = [0, 0.1, 0.2, 0.3, 0.4]

    t_max = 100
    n_t = 100
    alpha1 = np.linspace(0, 2.5, 100)
    alpha2 = np.linspace(0, np.pi, 100)

    n_workers = 80

    for radius in radii:
        graph, points = delaunay_with_holes(30, centres, [radius], n_nodes_hole=int(50 * radius))
        plt.figure()
        nx.draw(graph, pos=points, node_size=0.1)
        plt.savefig(f"delaunay_{radius}.pdf")

        Gsc = SimplicialComplex(graph=graph)

        scan_frustration_parameters(
            Gsc,
            filename=f"delaunay_one_hole_{radius}.pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            n_workers=n_workers,
            n_t=n_t,
            t_max=t_max,
            repeats=1,
        )
