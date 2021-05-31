import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters, plot_phases

matplotlib.use("Agg")


# ## Barbell 2 3-cliques

G = nx.Graph()

G.add_edge(0, 1, weight=1, edge_com=0)
G.add_edge(1, 2, weight=1, edge_com=0)
G.add_edge(2, 0, weight=1, edge_com=0)

G.add_edge(0, 3, weight=1, edge_com=1)

G.add_edge(3, 4, weight=1, edge_com=2)
G.add_edge(4, 5, weight=1, edge_com=2)
G.add_edge(5, 3, weight=1, edge_com=2)

node_com_dict = dict(zip(list(np.linspace(0, 5, 6).astype(int)), [0, 0, 0, 1, 1, 1]))
nx.set_node_attributes(G, node_com_dict, "node_com")

edge_community_assignment = np.array(list(nx.get_edge_attributes(G, "edge_com").values()))

Gsc = SimplicialComplex(graph=G, no_faces=False)

plt.figure()
nx.draw_networkx(G)


alpha1 = np.linspace(0, np.pi, 50)
alpha2 = np.linspace(0, np.pi, 50)
n_repeats = 1
n_workers = 12

scan_frustration_parameters(
    Gsc,
    filename="barbell_2_3.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


# the bar edge is 2
folder = "./results/"

path = folder + "barbell_2_3.pkl"
filename = "barbell_2_3.pdf"
plot_phases(path, filename)


Gsc.flip_edge_orientation(2)

n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="barbell_2_3_flip.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


# the bar edge is 2
folder = "./results/"

path = folder + "barbell_2_3_flip.pkl"
filename = "barbell_2_3_flip.pdf"
plot_phases(path, filename)


# ## Barbell 2 4-cliques


G = nx.Graph()

G.add_edge(0, 1, weight=1, edge_com=0)
G.add_edge(1, 2, weight=1, edge_com=0)
G.add_edge(2, 3, weight=1, edge_com=0)
G.add_edge(0, 3, weight=1, edge_com=0)
G.add_edge(0, 2, weight=1, edge_com=0)
G.add_edge(1, 3, weight=1, edge_com=0)

G.add_edge(0, 4, weight=1, edge_com=1)

G.add_edge(4, 5, weight=1, edge_com=2)
G.add_edge(5, 6, weight=1, edge_com=2)
G.add_edge(6, 7, weight=1, edge_com=2)
G.add_edge(4, 7, weight=1, edge_com=2)
G.add_edge(4, 6, weight=1, edge_com=2)
G.add_edge(5, 7, weight=1, edge_com=2)

node_com_dict = dict(zip(list(np.linspace(0, 7, 8).astype(int)), [0, 0, 0, 0, 1, 1, 1, 1]))
nx.set_node_attributes(G, node_com_dict, "node_com")

edge_community_assignment = np.array(list(nx.get_edge_attributes(G, "edge_com").values()))

Gsc = SimplicialComplex(graph=G, no_faces=False)

plt.figure()
nx.draw_networkx(G)


n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="barbell_2_4.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


# the bar edge is 3
folder = "./results/"

path = folder + "barbell_2_4.pkl"
filename = "barbell_2_4.pdf"
plot_phases(path, filename)


Gsc.flip_edge_orientation(3)

n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="barbell_2_4_flip.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


# the bar edge is 3
folder = "./results/"

path = folder + "barbell_2_4_flip.pkl"
filename = "barbell_2_4_flip.pdf"
plot_phases(path, filename)


# ## Two faces - common node


G = nx.Graph()

G.add_edge(0, 1, weight=1, edge_com=0)
G.add_edge(1, 2, weight=1, edge_com=0)
G.add_edge(2, 0, weight=1, edge_com=0)

G.add_edge(0, 3, weight=1, edge_com=1)
G.add_edge(3, 4, weight=1, edge_com=1)
G.add_edge(4, 0, weight=1, edge_com=1)

node_com_dict = dict(zip(list(np.linspace(0, 4, 5).astype(int)), [0, 0, 0, 0, 0]))
nx.set_node_attributes(G, node_com_dict, "node_com")

edge_community_assignment = np.array(list(nx.get_edge_attributes(G, "edge_com").values()))

Gsc = SimplicialComplex(graph=G, no_faces=False)

plt.figure()
nx.draw_networkx(G)


n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="2_faces_common_node.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


folder = "./results/"

path = folder + "2_faces_common_node.pkl"
filename = "2_faces_common_node.pdf"
plot_phases(path, filename)


# the flipped edge makes one triangle a directed cycle
Gsc.flip_edge_orientation(1)

n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="2_faces_common_node_dircycle.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


# the flipped edge makes one triangle a directed cycle
folder = "./results/"

path = folder + "2_faces_common_node_dircycle.pkl"
filename = "2_faces_common_node_dircycle.pdf"
plot_phases(path, filename)


# ## Two faces - common edge


G = nx.Graph()

G.add_edge(0, 1, weight=1, edge_com=0)
G.add_edge(1, 2, weight=1, edge_com=0)
G.add_edge(2, 3, weight=1, edge_com=0)
G.add_edge(3, 0, weight=1, edge_com=0)
G.add_edge(0, 2, weight=1, edge_com=0)

node_com_dict = dict(zip(list(np.linspace(0, 3, 4).astype(int)), [0, 0, 0, 0]))
nx.set_node_attributes(G, node_com_dict, "node_com")

edge_community_assignment = np.array(list(nx.get_edge_attributes(G, "edge_com").values()))

Gsc = SimplicialComplex(graph=G, no_faces=False)

plt.figure()
nx.draw_networkx(G)


n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="2_faces_common_edge.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


folder = "./results/"

path = folder + "2_faces_common_edge.pkl"
filename = "2_faces_common_edge.pdf"
plot_phases(path, filename)


# the flipped edge makes one triangle a directed cycle
Gsc.flip_edge_orientation(2)

n_repeats = 1

scan_frustration_parameters(
    Gsc,
    filename="2_faces_common_edge_flip.pkl",
    alpha1=alpha1,
    alpha2=alpha2,
    repeats=n_repeats,
    n_workers=n_workers,
)


folder = "./results/"

path = folder + "2_faces_common_edge_flip.pkl"
filename = "2_faces_common_edge_flip.pdf"
plot_phases(path, filename)
