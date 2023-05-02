"""Example of nonlinear simplicial kuramoto model."""
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.measures import compute_order_parameter

from simplicial_kuramoto.graph_generator import make_simple

if __name__ == "__main__":
    t_max = 500
    n_t = 1000
    np.random.seed(42)

    sigmas = np.linspace(0.2, 0.7, 50)
    epsilons = np.linspace(0, 1, 10)

    sc = make_simple(plot=True)
    plt.savefig("sc.pdf")
    plt.close()

    phase_init = np.random.uniform(0, np.pi, 5)
    alpha_1 = np.random.uniform(0, 1, 5)

    g_f = Path("global_orders_scan.csv")
    n_f = Path("node_orders_scan.csv")
    f_f = Path("face_orders_scan.csv")
    if not g_f.exists():
        global_orders = pd.DataFrame()
        node_orders = pd.DataFrame()
        face_orders = pd.DataFrame()

        for epsilon in epsilons:
            print('epsilon:', epsilon)
            for sigma in tqdm(sigmas):
                edge_res = integrate_edge_kuramoto(
                    sc,
                    phase_init,
                    t_max,
                    n_t,
                    alpha_1=alpha_1,
                    alpha_2=0.0,
                    sigma_up=sigma,
                    sigma_down=sigma,
                    variant="nonlinear",
                    variant_params={"epsilon": epsilon},
                    disable_tqdm=True,
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
