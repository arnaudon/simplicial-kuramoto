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

    t_max = 50
    n_t = 10000
    n_workers = 80
    initial_phase = np.random.normal(0, 1, Gsc.num_nodes + Gsc.n_edges)
    res = integrate_tower_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_0=0.5,
        alpha_1=0.5,
        alpha_2=np.pi / 2 - 0.05,
        sigma_0=1.0,
        sigma_1=1.0,
    )
    n_min = 1000
    res.t = res.t[n_min:]
    res.y = res.y[:, n_min:]
    node_order = compute_node_order_parameter(res.y[: Gsc.num_nodes], Gsc)
    edge_order = compute_order_parameter(res.y[Gsc.num_nodes :], Gsc)[0]

    plt.figure()
    plt.plot(res.t, node_order, c="k", label="node order")
    plt.plot(res.t, edge_order, c="r", label="edge order")
    plt.axhline(1, ls="--", c="k")
    plt.legend()

    plt.figure(figsize=(5, 5))
    res.y = np.sin(res.y)
    # plt.plot(res.t, res.y[0], c="k", label='node 0')
    # plt.plot(res.t, res.y[1], c="k", label='node 1')
    # plt.plot(res.t, res.y[2], c="k", label='node 2')
    plt.scatter(res.y[0], res.y[1], c=res.y[2], marker="+")
    plt.axis([-1, 1, -1, 1])
    plt.legend()
    plt.suptitle("node")
    plt.savefig("node.pdf")

    plt.figure(figsize=(5, 5))
    # plt.plot(res.t, res.y[3], c="r", label='edge 0')
    # plt.plot(res.t, res.y[4], c="r", label='edge 0')
    # plt.plot(res.t, res.y[5], c="r", label='edge 0')
    plt.scatter(res.y[3], res.y[4], c=res.y[5], marker="+")
    plt.axis([-1, 1, -1, 1])
    plt.suptitle("edge")
    plt.legend()

    plt.savefig("edges.pdf")
    plt.show()
