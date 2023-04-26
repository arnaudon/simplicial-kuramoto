"""Example of fustrated simplicial kuramoto model."""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.measures import compute_order_parameter
from simplicial_kuramoto.measures import (
    compute_critical_couplings,
    compute_necessary_bounds,
    compute_sufficient_bounds,
)

from simplicial_kuramoto.graph_generator import make_simple


if __name__ == "__main__":
    t_max = 5000
    n_t = 1000
    sigmas = np.linspace(0.05, 1.0, 100)

    np.random.seed(42)

    sc = make_simple(plot=True)
    plt.savefig('sc.pdf')
    plt.close()

    phase_init = np.random.uniform(0, np.pi, 5)
    alpha_1 = np.random.uniform(0, 1, 5)
    alpha_2 = 0.0

    sigma_down, sigma_up = compute_necessary_bounds(sc, alpha_1)
    sigma_down_critical, sigma_up_critical = compute_critical_couplings(sc, alpha_1)
    bounds_down, bounds_up = compute_sufficient_bounds(sc, alpha_1)

    o = []
    n = []
    f = []
    for sigma in tqdm(sigmas):
        edge_res = integrate_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_up=1,
            sigma_down=sigma,
            disable_tqdm=True,
        )

        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        o.append(np.mean(order[-200:]))
        n.append(np.mean(node_order[-200:]))
        f.append(np.mean(face_order[-200:]))

    plt.figure()
    plt.plot(sigmas, o, label="order", c="k")
    plt.plot(sigmas, n, label="node", c="g", ls="--")
    plt.axvline(sigma_down, label="down", c="k")
    plt.axvline(sigma_down_critical, label="down critical", c="r")
    plt.axvline(bounds_down, label="bound down", c="m")
    plt.xlabel('sigma down')
    plt.ylabel('order parameter')
    plt.legend()
    plt.savefig("stability_sigma_down.pdf")

    o = []
    n = []
    f = []
    for sigma in tqdm(sigmas):
        edge_res = integrate_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_up=sigma,
            sigma_down=1.0,
            disable_tqdm=True,
        )
        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        o.append(np.mean(order[-200:]))
        n.append(np.mean(node_order[-200:]))
        f.append(np.mean(face_order[-200:]))

    plt.figure()
    plt.plot(sigmas, o, label="order", c="k")
    plt.plot(sigmas, f, label="face", c="b", ls="--")
    plt.axvline(sigma_up, label="up", c="k")
    plt.axvline(sigma_up_critical, label="up critical", c="r")
    plt.axvline(bounds_up, label="bound up", c="m")
    plt.xlabel('sigma up')
    plt.ylabel('order parameter')
    plt.legend()
    plt.savefig("stability_sigma_up.pdf")
