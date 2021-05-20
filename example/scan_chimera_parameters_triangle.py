import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from chimera_scan import scan_chimera_parameters


G = nx.Graph()
G.add_edge(0, 1, weight=1, edge_com=0)
G.add_edge(1, 2, weight=1, edge_com=0)
G.add_edge(2, 0, weight=1, edge_com=0)

edge_community_assignment = np.array(list(nx.get_edge_attributes(G, "edge_com").values()))


Gsc = SimplicialComplex(graph=G, no_faces=False)
Gsc_noface = SimplicialComplex(graph=G, no_faces=True)
Gsc.flip_edge_orientation(1)
Gsc_noface.flip_edge_orientation(1)
print(Gsc.B0.toarray())
print(Gsc.B1.toarray())

#alpha1 = np.linspace(np.pi/4.0, np.pi/2.0, 15)
#alpha2 = np.linspace(np.pi/4.0, np.pi/2.0, 15)
alpha1 = np.linspace(0, 2., 15)
alpha2 = np.linspace(0, np.pi, 40)
print(alpha1, alpha2)
n_repeats = 1
"""
results = scan_chimera_parameters(
    Gsc,
    filename="triangle_face_flip.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=12,
)

results = scan_chimera_parameters(
    Gsc_noface,
    filename="triangle_noface_flip.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=12,
)
"""

Gsc.flip_edge_orientation(0)
Gsc_noface.flip_edge_orientation(0)

print(Gsc.B0.toarray())
print(Gsc.B1.toarray())
results = scan_chimera_parameters(
    Gsc,
    filename="triangle_face.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=12,
)

results = scan_chimera_parameters(
    Gsc_noface,
    filename="triangle_noface.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=12,
)
