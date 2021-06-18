import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

if __name__ == "__main__":

    G = nx.Graph()
    # one face 0, 1, 2
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(0, 2, weight=1, edge_com=0)

    # middle edge
    G.add_edge(1, 2, weight=2, edge_com=0)

    # other face, 1, 2, 3
    G.add_edge(2, 3, weight=1, edge_com=0)
    G.add_edge(1, 3, weight=1, edge_com=0)

    alpha1 = np.linspace(0, 2.2, 100)
    alpha2 = np.linspace(0, np.pi, 100)
    n_repeats = 1
    t_max = 100
    n_t = 100
    n_workers = 80

    Gsc = SimplicialComplex(graph=G, no_faces=False)
    # flip so that each face is oriented
    Gsc.flip_edge_orientation(1)
    Gsc.flip_edge_orientation(4)

    scan_frustration_parameters(
        Gsc,
        filename="2_triangles_face.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
    )

    # flip one edge not middle
    Gsc_flip = SimplicialComplex(graph=G)  # , faces=[[0, 1, 2], [1, 3, 2]])
    Gsc_flip.flip_edge_orientation(1)
    # Gsc_flip.flip_edge_orientation(4)

    scan_frustration_parameters(
        Gsc_flip,
        filename="2_triangles_face_flip.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
    )

    # flip middle edge
    Gsc_flip = SimplicialComplex(graph=G)
    Gsc_flip.flip_edge_orientation(1)
    Gsc_flip.flip_edge_orientation(4)
    Gsc_flip.flip_edge_orientation(2)

    scan_frustration_parameters(
        Gsc_flip,
        filename="2_triangles_face_flip_middle.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=n_workers,
        t_max=t_max,
        n_t=n_t,
    )

    """
    # remove one face
    Gsc_flip = SimplicialComplex(graph=G, faces=[[0, 1, 2]])
    Gsc_flip.flip_edge_orientation(1)
    Gsc_flip.flip_edge_orientation(4)

    scan_frustration_parameters(
        Gsc_flip,
        filename="2_triangles_face_one_face.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=12,
        t_max=t_max,
        n_t=n_t,
    )

    # remove one face flip one edge in hloe
    Gsc_flip = SimplicialComplex(graph=G, faces=[[0, 1, 2]])
    Gsc_flip.flip_edge_orientation(1)
    # Gsc_flip.flip_edge_orientation(4)

    scan_frustration_parameters(
        Gsc_flip,
        filename="2_triangles_face_one_face_flip.pkl",
        alpha1=alpha1,
        alpha2=alpha2,
        repeats=n_repeats,
        n_workers=12,
        t_max=t_max,
        n_t=n_t,
    )
    """
