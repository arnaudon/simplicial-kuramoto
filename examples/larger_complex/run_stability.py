"""Example of fustrated simplicial kuramoto model."""
import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.measures import compute_order_parameter
from simplicial_kuramoto.measures import (
    compute_critical_couplings,
    compute_necessary_bounds,
    compute_sufficient_bounds,
)

from simplicial_kuramoto import SimplicialComplex

from simplicial_kuramoto.graph_generator import delaunay_with_holes

if __name__ == "__main__":
    t_max = 1000
    n_t = 1000
    n_workers = 10
    repeats = 10
    n_alpha2 = 200
    np.random.seed(2)

    centres = [[0.3, 0.3], [0.7, 0.7]]
    radii = [0.15, 0.1]

    graph, points = delaunay_with_holes(50, centres, radii, n_nodes_hole=7)
    sc = SimplicialComplex(graph=graph)

    phase_init = np.random.uniform(0, np.pi, sc.n_edges)
    alpha_1 = np.random.uniform(0, 1, sc.n_edges)
    alpha_2 = 0.0

    sigma_down, sigma_up = compute_necessary_bounds(sc, alpha_1, alpha_2)
    print(sigma_down, sigma_up)
    sigma_down_critical, sigma_up_critical = compute_critical_couplings(sc, alpha_1, alpha_2)
    print(sigma_down_critical, sigma_up_critical)
    bounds_down, bounds_up = compute_sufficient_bounds(sc, alpha_1, alpha_2)
    print(bounds_down, bounds_up)
    sigmas = np.linspace(0.05, 3.5, 20)
    o = []
    n = []
    f = []
    for sigma in sigmas:
        edge_res = integrate_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_up=1,
            sigma_down=sigma,
        )
        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        o.append(np.mean(order[-200:]))
        n.append(np.mean(node_order[-200:]))
        f.append(np.mean(face_order[-200:]))

    plt.figure()
    plt.plot(sigmas, o, label="order", c="k")
    plt.plot(sigmas, n, label="node", c="g", ls="--")
    plt.plot(sigmas, f, label="face", c="b", ls="--")
    plt.axvline(sigma_down, label="down", c="k")
    plt.axvline(sigma_down_critical, label="down critical", c="r")
    plt.axvline(bounds_down, label="bound down", c="m")
    plt.legend()
    plt.savefig("stability_sigma_down.pdf")

    o = []
    n = []
    f = []
    for sigma in sigmas:
        edge_res = integrate_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_up=sigma,
            sigma_down=1.0,
        )
        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        o.append(np.mean(order[-200:]))
        n.append(np.mean(node_order[-200:]))
        f.append(np.mean(face_order[-200:]))

    plt.figure()
    plt.plot(sigmas, o, label="order", c="k")
    plt.plot(sigmas, n, label="node", c="g", ls="--")
    plt.plot(sigmas, f, label="face", c="b", ls="--")
    plt.axvline(sigma_up, label="up", c="k")
    plt.axvline(sigma_up_critical, label="up critical", c="r")
    plt.axvline(bounds_up, label="bound up", c="m")
    plt.legend()
    plt.savefig("stability_sigma_up.pdf")
