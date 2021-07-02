import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import delaunay_with_holes
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import (
    get_subspaces,
    proj_subspace,
    compute_simplicial_order_parameter,
)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    centres = [[0.25, 0.25], [0.75, 0.75]]

    t_max = 100
    n_t = 100

    n_workers = 80

    radius = 0.1
    graph, points = delaunay_with_holes(
        30, centres, [radius, radius], n_nodes_hole=int(50 * radius)
    )
    Gsc = SimplicialComplex(graph=graph)

    t_max = 1000
    n_t = 1000
    n_min = 0
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    alpha_1 = 2.0
    harm_alpha = np.array([1.0, 1.0])

    alpha_1 = alpha_1 * harm_subspace.dot(harm_alpha)
    initial_phase = alpha_1  # np.random.random(Gsc.n_edges)

    plt.figure(figsize=(4, 3))
    for alpha_2 in [1.45]:  # , 0.45, 0.5, 0.7, 1.0, 1.4, 1.5]:
        print("alpha_2=", alpha_2)
        res = integrate_edge_kuramoto(
            Gsc,
            initial_phase,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
        )
        result = res.y[:, n_min:]
        time = res.t[n_min:]

        global_order, partial_orders = compute_simplicial_order_parameter(result, harm_subspace)
        plt.plot(time, global_order, label=f"alpha_2 = {alpha_2}")
        for partial_order in partial_orders:
            plt.plot(time, partial_order, label=f"partial, alpha_2 = {alpha_2}")
    plt.gca().set_xlim(time[0], time[-1])
    plt.axhline(1.0, ls="--", c="k")
    plt.legend(loc="best")
    # plt.gca().set_ylim(0, 1.02)
    plt.xlabel("time")
    plt.ylabel("order parameter")
    plt.legend(loc="best")
    plt.savefig(f"scan_order.pdf", bbox_inches="tight")
