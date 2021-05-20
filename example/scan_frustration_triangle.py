import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

if __name__ == "__main__":

    G = nx.Graph()
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(2, 0, weight=1, edge_com=0)

    Gsc = SimplicialComplex(graph=G, no_faces=False)
    Gsc_noface = SimplicialComplex(graph=G, no_faces=True)

    Gsc.flip_edge_orientation(1)
    Gsc_noface.flip_edge_orientation(1)

    alpha1 = np.linspace(0, 2.0, 15)
    alpha2 = np.linspace(0, np.pi, 40)
    n_repeats = 1

    scan_frustration_parameters(
        Gsc,
        filename="triangle_face_flip.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=12,
    )

    scan_frustration_parameters(
        Gsc_noface,
        filename="triangle_noface_flip.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=12,
    )

    Gsc.flip_edge_orientation(0)
    Gsc_noface.flip_edge_orientation(0)

    scan_frustration_parameters(
        Gsc,
        filename="triangle_face.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=12,
    )

    scan_frustration_parameters(
        Gsc_noface,
        filename="triangle_noface.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=12,
    )
