import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_chimera_parameters


G = nx.Graph()
G.add_edge(0, 1, weight=1, edge_com=0)
G.add_edge(1, 2, weight=1, edge_com=0)
G.add_edge(2, 0, weight=1, edge_com=0)
G.add_edge(3, 4, weight=1, edge_com=1)
G.add_edge(4, 5, weight=1, edge_com=1)
G.add_edge(5, 3, weight=1, edge_com=1)
G.add_edge(2, 3, weight=1, edge_com=2)

node_com_dict = dict(zip(list(np.linspace(0, 5, 6).astype(int)), [0, 0, 0, 1, 1, 1]))
nx.set_node_attributes(G, node_com_dict, "node_com")
edge_community_assignment = np.array(list(nx.get_edge_attributes(G, "edge_com").values()))


Gsc = SimplicialComplex(graph=G, no_faces=False)
Gsc_noface = SimplicialComplex(graph=G, no_faces=True)

alpha1 = np.linspace(0, np.pi, 15)
alpha2 = np.linspace(0, np.pi, 15)
n_repeats = 1

results = scan_chimera_parameters(
    Gsc,
    filename="two_comms_three_nodes.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=12,
)
"""
results = scan_chimera_parameters(
    Gsc_noface,
    filename="two_comms_three_nodes_nofaces.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=12,
)
"""
