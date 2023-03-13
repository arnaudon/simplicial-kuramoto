"""Example of nonlinear simplicial kuramoto model."""
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simplicial_kuramoto.integrators import (
    integrate_nonlinear_edge_kuramoto,
    integrate_edge_kuramoto,
)
from simplicial_kuramoto.plotting import plot_edge_kuramoto
from simplicial_kuramoto.measures import compute_order_parameter

from make_sc import make_sc

if __name__ == "__main__":
    t_max = 1000
    n_t = 500
    np.random.seed(42)

    sc = make_sc()
    phase_init = np.random.uniform(0, np.pi, 5)
    alpha_1 = np.random.uniform(0, 1, 5)
    edge_res = integrate_nonlinear_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_1=alpha_1,
        alpha_2=0.0,
        sigma_up=0.2,
        sigma_down=0.2,
        epsilon=0.1,
    )
    plot_edge_kuramoto(edge_res)
    plt.savefig("nonlinear_edge_kuramoto.pdf")

    plt.figure()
    order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
    plt.plot(edge_res.t, order, label="global")
    plt.plot(edge_res.t, node_order, label="node")
    plt.plot(edge_res.t, face_order, label="face")
    plt.legend(loc="best")
    plt.savefig("nonlinear_order_edge.pdf")

    sigmas = np.linspace(0.1, 0.9, 20)
    epsilons = np.linspace(0, 1, 20)

    g_f = Path("global_orders_scan.csv")
    n_f = Path("node_orders_scan.csv")
    f_f = Path("face_orders_scan.csv")
    if not g_f.exists():
        global_orders = pd.DataFrame()
        node_orders = pd.DataFrame()
        face_orders = pd.DataFrame()
        for epsilon in epsilons:

            for sigma in sigmas:
                edge_res = integrate_nonlinear_edge_kuramoto(
                    sc,
                    phase_init,
                    t_max,
                    n_t,
                    alpha_1=alpha_1,
                    alpha_2=0.0,
                    sigma_up=sigma,
                    sigma_down=sigma,
                    epsilon=epsilon,
                )
                order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
                global_orders.loc[epsilon, sigma] = np.mean(order[-200:])
                node_orders.loc[epsilon, sigma] = np.mean(node_order[-200:])
                face_orders.loc[epsilon, sigma] = np.mean(face_order[-200:])

        global_orders.to_csv(g_f)
        node_orders.to_csv(n_f)
        face_orders.to_csv(f_f)
    else:
        global_orders = pd.read_csv(g_f)
        node_orders = pd.read_csv(n_f)
        face_orders = pd.read_csv(f_f)

    plt.figure()
    sns.heatmap(global_orders)
    plt.tight_layout()
    plt.savefig("nonlinear_coupling_scan_global.pdf")

    plt.figure()
    sns.heatmap(node_orders)
    plt.tight_layout()
    plt.savefig("nonlinear_coupling_scan_node.pdf")

    plt.figure()
    sns.heatmap(face_orders)
    plt.tight_layout()
    plt.savefig("nonlinear_coupling_scan_face.pdf")
