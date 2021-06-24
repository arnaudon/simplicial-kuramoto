import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import delaunay_with_holes

if __name__ == "__main__":
    np.random.seed(42)
    graph, points = delaunay_with_holes(6)
    alpha1 = np.linspace(0, 2.2, 100)
    alpha2 = np.linspace(0, np.pi, 100)
    n_repeats = 1

    t_max = 2000
    n_t = 100
    n_workers = 80

    plt.figure()
    nx.draw(graph, pos=points)#, node_size=0.1)
    nx.draw_networkx_labels(graph, pos=points)
    plt.savefig(f"graph.pdf")

    Gsc = SimplicialComplex(graph=graph)
    f = Gsc.faces
    print(f)
    del f[1]
    print(f)
    Gsc = SimplicialComplex(graph=graph, faces=f)

    scan_frustration_parameters(
        Gsc,
        filename="2_triangles_face.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )
