"""Example of tower simplicial kuramoto model."""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.plotting import plot_node_kuramoto, plot_edge_kuramoto
from simplicial_kuramoto.integrators import integrate_tower_kuramoto
from simplicial_kuramoto.integrators import compute_order_parameter, compute_node_order_parameter

from make_sc import make_sc

if __name__ == "__main__":
    t_max = 500
    n_t = 500
    n_workers = 10
    np.random.seed(42)

    sc = make_sc()
    phase_init = np.random.uniform(0, 1.0, 5 + 4)
    alpha_1 = np.random.uniform(0, 1, 5)

    res = integrate_tower_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=0.0,
        alpha_1=0.0,
        alpha_2=0.2,
        sigma_0=1.,
        sigma_1=1.,
    )
    node_res = deepcopy(res)
    node_res.y = res.y[: sc.num_nodes]
    edge_res = deepcopy(res)
    edge_res.y = res.y[sc.num_nodes :]

    node_order = compute_node_order_parameter(sc, node_res.y)
    edge_order = compute_order_parameter(sc, edge_res.y)[0]

    plt.figure()
    plot_node_kuramoto(node_res)
    plt.savefig("tower_node.pdf")

    plt.figure()
    plot_edge_kuramoto(edge_res)
    plt.savefig("tower_edges.pdf")

    plt.figure()
    plt.plot(res.t, node_order, label="node")
    plt.plot(res.t, edge_order, label="edge global")
    plt.legend(loc="best")
    plt.savefig("tower_order.pdf")
