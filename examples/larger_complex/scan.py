"""Scan frustration parameter."""
import numpy as np
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

from simplicial_kuramoto.graph_generator import delaunay_with_holes
from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.plotting import draw_simplicial_complex

if __name__ == "__main__":
    np.random.seed(42)

    t_max = 1000
    n_t = 500
    n_workers = 10
    repeats = 10
    n_alpha2 = 200

    centres = [[0.3, 0.3], [0.7, 0.7]]
    radii = [0.15, 0.1]

    graph, points = delaunay_with_holes(50, centres, radii, n_nodes_hole=7)
    Gsc = SimplicialComplex(graph=graph)
    draw_simplicial_complex(Gsc, filename="larger_graph.pdf", with_labels=False)

    scan_frustration_parameters(
        Gsc,
        folder="./results/",
        filename="complex.pkl",
        alpha1=[0.0],
        alpha2=np.linspace(0, np.pi / 2.0, n_alpha2),
        repeats=repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
        harmonic=True,
    )
