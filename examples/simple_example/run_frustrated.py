"""Example of fustrated simplicial kuramoto model."""
import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.integrators import integrate_node_kuramoto
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import (
    compute_node_order_parameter,
    compute_order_parameter,
)
from simplicial_kuramoto.plotting import plot_node_kuramoto, plot_edge_kuramoto
from make_sc import make_sc

if __name__ == "__main__":
    t_max = 200
    n_t = 500
    n_workers = 10
    repeats = 10
    n_alpha2 = 200
    np.random.seed(42)

    sc= make_sc()
    phase_init = np.random.uniform(0, np.pi, 4)

    node_res = integrate_node_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=np.random.uniform(0, 1, 4),
        alpha_1=0.5,
        sigma=0.2,
    )

    plt.figure()
    plt.plot(node_res.t, compute_node_order_parameter(sc, node_res.y))
    plt.savefig("order_node.pdf")

    plt.figure()
    plot_node_kuramoto(node_res)
    plt.savefig("node_kuramoto.pdf")
    plt.close()

    phase_init = np.random.uniform(0, np.pi, 5)

    edge_res = integrate_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_1=np.random.uniform(0, 1, 5),
        alpha_2=0.5,
        sigma=0.2,
    )
    plot_edge_kuramoto(edge_res)
    plt.savefig("edge_kuramoto.pdf")

    plt.figure()
    order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
    plt.plot(edge_res.t, order, label="global")
    plt.plot(edge_res.t, node_order, label="node")
    plt.plot(edge_res.t, face_order, label="face")
    plt.legend(loc="best")
    plt.savefig("order_edge.pdf")

    plt.close()
