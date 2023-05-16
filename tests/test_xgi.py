import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.simplicial_complex import xgi_to_internal
import xgi


def make_sc_internal():

    G = nx.Graph()

    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(0, 3, weight=1, edge_com=0)
    G.add_edge(1, 3, weight=1, edge_com=0)
    G.add_edge(1, 4, weight=1, edge_com=0)
    G.add_edge(2, 4, weight=1, edge_com=0)

    G.add_edge(3, 5, weight=1, edge_com=0)
    G.add_edge(3, 6, weight=1, edge_com=0)
    G.add_edge(4, 6, weight=1, edge_com=0)
    G.add_edge(4, 7, weight=1, edge_com=0)
    G.add_edge(5, 6, weight=1, edge_com=0)
    G.add_edge(6, 7, weight=1, edge_com=0)

    G.add_edge(1, 6, weight=1, edge_com=0)

    # pos = nx.spring_layout(G,)
    pos_ = {}
    pos_[0] = np.array([0, 0])
    pos_[1] = np.array([1, 0])
    pos_[2] = np.array([2, 0])
    pos_[3] = np.array([0, 1])
    pos_[4] = np.array([2, 1])
    pos_[5] = np.array([0, 2])
    pos_[6] = np.array([1, 2])
    pos_[7] = np.array([2, 2])

    for n in G.nodes:
        G.nodes[n]["pos"] = pos_[n]

    sc = SimplicialComplex(graph=G, no_faces=False)

    del sc.faces[3]
    sc.n_faces -= 1
    return sc


def make_sc_xgi(sc):
    sc_xgi = xgi.SimplicialComplex([list(e) for e in sc.graph.edges])
    sc_xgi.add_simplices_from(sc.faces)
    return sc_xgi


def test_xgi_to_internal():
    sc = make_sc_internal()
    sc_xgi = make_sc_xgi(sc)
    _sc_xgi = xgi_to_internal(sc_xgi)

    np.testing.assert_array_equal(sc.W2.toarray(), _sc_xgi.W2.toarray())

    np.testing.assert_array_equal(sc.N0.toarray(), _sc_xgi.N0.toarray())
    np.testing.assert_array_equal(sc.N0s.toarray(), _sc_xgi.N0s.toarray())
    np.testing.assert_array_equal(sc.lifted_N0.toarray(), _sc_xgi.lifted_N0.toarray())
    np.testing.assert_array_equal(sc.lifted_N0sn.toarray(), _sc_xgi.lifted_N0sn.toarray())
    np.testing.assert_array_equal(sc.N1.toarray(), _sc_xgi.N1.toarray())
    np.testing.assert_array_equal(sc.lifted_N1.toarray(), _sc_xgi.lifted_N1.toarray())
    np.testing.assert_array_equal(sc.lifted_N1sn.toarray(), _sc_xgi.lifted_N1sn.toarray())
