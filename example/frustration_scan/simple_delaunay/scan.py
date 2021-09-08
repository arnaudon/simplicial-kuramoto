import numpy as np
import sys
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.plotting import draw_simplicial_complex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import delaunay_with_holes

def make_delaunay():
    np.random.seed(42)
    points = [[x, y] for x in np.linspace(0, 1, 3) for y in np.linspace(0, 1, 3)]
    p_list = [8, 6, 2, 0]
    points = [p for i, p in enumerate(points) if i not in p_list[:2]]
    graph, points = delaunay_with_holes(points=points)
    return graph, points

if __name__ == "__main__":

    graph, points = make_delaunay()
    alpha1 = [1.0]
    alpha2 = np.linspace(0, np.pi / 2.0, 100)
    n_repeats = 2

    t_max = 1000
    n_t = 1000
    n_workers = 80
    Gsc = SimplicialComplex(graph=graph)
    edge_flip = [0, 10, 7]
    Gsc.flip_edge_orientation(edge_flip)

    draw_simplicial_complex(Gsc, filename="simplicial_complex.pdf")

    Gsc = SimplicialComplex(graph=graph)
    faces = Gsc.faces

    for face in range(len(faces)):
        f = copy(faces)
        del f[face]
        Gsc = SimplicialComplex(graph=graph, faces=f)
        Gsc.flip_edge_orientation(edge_flip)
        draw_simplicial_complex(Gsc, filename=f"simplicial_complex_face_{face}.pdf")

        scan_frustration_parameters(
            Gsc,
            filename=f"result_{face}.pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=n_workers,
            t_max=t_max,
            n_t=n_t,
            harmonic=True,
        )
