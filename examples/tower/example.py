import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.integrators import integrate_tower_kuramoto
from simplicial_kuramoto.frustration_scan import (
    compute_order_parameter,
    compute_node_order_parameter,
)

if __name__ == "__main__":
    np.random.seed(42)
    G = nx.Graph()
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(2, 0, weight=1, edge_com=0)

    Gsc = SimplicialComplex(graph=G)
    Gsc.flip_edge_orientation([0, 1])

    t_max = 100
    n_t = 4000
    n_workers = 80
    initial_phase = np.random.normal(0, 1, Gsc.n_nodes + Gsc.n_edges)
    res = integrate_tower_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_0=0*np.random.normal(0, 1, Gsc.n_nodes),
        alpha_1=np.random.normal(0, 1, Gsc.n_edges),
        alpha_2=0.0,
        sigma_0=1.0,
        sigma_1=0.5,
    )
    n_min = 100
    res.t = res.t[n_min:]
    res.y = res.y[:, n_min:]
    node_order = compute_node_order_parameter(res.y[:Gsc.n_nodes], Gsc)
    edge_order = compute_order_parameter(res.y[Gsc.n_nodes:], Gsc)[0]

    plt.figure()
    plt.plot(res.t, node_order, c='k')
    plt.plot(res.t, edge_order, c='r')
    print(node_order)
    plt.figure(figsize=(10, 3))
    res.y = np.sin(res.y)
    plt.plot(res.t, res.y[0], c="k")
    plt.plot(res.t, res.y[1], c="k")
    plt.plot(res.t, res.y[2], c="k")
    plt.savefig("node.pdf")

    plt.figure(figsize=(10, 3))
    plt.plot(res.t, res.y[3], c="r")
    plt.plot(res.t, res.y[4], c="r")
    plt.plot(res.t, res.y[5], c="r")

    plt.savefig("edges.pdf")
    plt.show()
