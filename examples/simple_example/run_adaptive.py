"""Example of adaptative simplicial kuramoto model."""
import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.integrators import (
    integrate_adaptive_edge_kuramoto,
    integrate_edge_kuramoto,
)
from simplicial_kuramoto.plotting import plot_edge_kuramoto
from simplicial_kuramoto.frustration_scan import compute_order_parameter

from make_sc import make_sc

if __name__ == "__main__":
    t_max = 1000
    n_t = 500
    n_workers = 10
    repeats = 10
    n_alpha2 = 200
    np.random.seed(42)

    sc = make_sc()
    phase_init = np.random.uniform(0, np.pi, 5)
    alpha_1 = np.random.uniform(0, 1, 5)
    edge_res = integrate_adaptive_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_1=alpha_1,
        alpha_2=0.0,
        sigma=0.2,
    )
    plot_edge_kuramoto(edge_res)
    plt.savefig("adaptive_edge_kuramoto.pdf")

    plt.figure()
    order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
    plt.plot(edge_res.t, order, label="global")
    plt.plot(edge_res.t, node_order, label="node")
    plt.plot(edge_res.t, face_order, label="face")
    plt.legend(loc="best")
    plt.savefig("adaptive_order_edge.pdf")

    sigmas = np.linspace(0.1, 0.9, 50)
    global_orders = []
    node_orders = []
    face_orders = []

    adapt_global_orders = []
    adapt_node_orders = []
    adapt_face_orders = []
    for sigma in sigmas:
        edge_res = integrate_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=0.0,
            sigma=sigma,
        )
        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        global_orders.append(np.mean(order[-200:]))
        node_orders.append(np.mean(node_order[-200:]))
        face_orders.append(np.mean(face_order[-200:]))

        edge_res = integrate_adaptive_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=0.0,
            sigma=sigma,
        )
        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        adapt_global_orders.append(np.mean(order[-200:]))
        adapt_node_orders.append(np.mean(node_order[-200:]))
        adapt_face_orders.append(np.mean(face_order[-200:]))

    plt.figure()
    plt.plot(sigmas, global_orders, label="global", c="C0")
    plt.plot(sigmas, node_orders, label="node", c="C1")
    plt.plot(sigmas, face_orders, label="face", c="C2")

    plt.plot(sigmas, adapt_global_orders, label="adapt global", ls="--", c="C0")
    plt.plot(sigmas, adapt_node_orders, label="adapt node", ls="--", c="C1")
    plt.plot(sigmas, adapt_face_orders, label="adapt face", ls="--", c="C2")
    plt.legend()
    plt.savefig("adaptive_coupling_scan.pdf")
