import numpy as np
import yaml
import networkx as nx
import random
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import (
    integrate_edge_kuramoto,
    get_subspaces,
    compute_harmonic_projections,
)
from simplicial_kuramoto.graph_generator import delaunay_with_holes


if __name__ == "__main__":

    np.random.seed(42)
    centres = [[0.25, 0.25], [0.75, 0.75]]
    radius = 0.15

    t_max = 500
    n_t = 500

    n_workers = 80

    graph, points = delaunay_with_holes(
        30, centres, [radius, radius], n_nodes_hole=int(50 * radius)
    )
    Gsc = SimplicialComplex(graph=graph)

    n_min = int(0.9 * n_t)
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    As = np.linspace(0, np.pi/2, 20)
    partials = {}
    for alpha_2 in [0.0, 0.2, 0.5, 1.0, 1.5]:
        for a in As:
            harm_alpha = np.array([np.cos(a), np.sin(a)])
            alpha_1 = harm_subspace.dot(harm_alpha)
            # initial_phase = np.random.random(len(alpha_1))  # alpha_1
            initial_phase = alpha_1

            res = integrate_edge_kuramoto(
                Gsc, initial_phase, t_max, n_t, alpha_1=alpha_1, alpha_2=alpha_2
            )
            result = res.y[:, n_min:]
            time = res.t[n_min:]
            partial_orders = compute_harmonic_projections(result, harm_subspace)
            partials[a] = np.mean(partial_orders, axis=1)
            yaml.dump(partials, open("partial.yaml", "w"))

        plt.figure(figsize=(4, 3))
        plt.plot(partials.keys(), [x[0] for x in partials.values()])
        plt.plot(partials.keys(), [x[1] for x in partials.values()])
        plt.axvline(np.pi/4, c="k", ls="--")
        plt.axhline(0.5, c="k", ls="--")
        plt.xlabel("alpha1 angle")
        plt.ylabel("partial orders")
        plt.axis([0, np.pi/2, 0, 1])
        plt.savefig(f"partials_alpha1_scan_{alpha_2}.pdf", bbox_inches='tight')
        plt.close()
