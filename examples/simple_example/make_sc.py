import numpy as np
import networkx as nx
import xgi
import matplotlib.pyplot as plt

from simplicial_kuramoto.plotting import draw_simplicial_complex

from simplicial_kuramoto import SimplicialComplex


def make_sc_internal():

    G = nx.Graph()

    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=1)
    G.add_edge(2, 0, weight=1)
    G.add_edge(0, 3, weight=1)
    G.add_edge(1, 3, weight=1)

    # pos = nx.spring_layout(G,)
    pos_ = {}
    pos_[0] = np.array([0, 0])
    pos_[1] = np.array([0, 1])
    pos_[2] = np.array([1, 0.5])
    pos_[3] = np.array([-1, 0.5])

    for n in G.nodes:
        G.nodes[n]["pos"] = pos_[n]

    sc = SimplicialComplex(graph=G, faces=[[0, 1, 2]])

    draw_simplicial_complex(sc, filename="sc.pdf")
    plt.close()
    return sc


def make_sc():
    sc = make_sc_internal()
    pos = [sc.graph.nodes[n]["pos"] for n in sc.graph.nodes]
    sc_xgi = xgi.SimplicialComplex([list(e) for e in sc.graph.edges])
    sc_xgi.add_simplices_from(sc.faces)

    plt.figure()
    xgi.draw(sc_xgi, pos=pos)
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.savefig("sc.pdf")
    plt.close()

    return sc_xgi
