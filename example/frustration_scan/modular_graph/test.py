import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import modular_graph
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import (
    get_subspaces,
    proj_subspace,
    compute_simplicial_order_parameter,
    compute_harmonic_projections,
)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    n_workers = 80
    n_clusters = 3
    graph = modular_graph(n_clusters, 7, 2, inter_weight=0.4, intra_weight=0.6)
    Gsc = SimplicialComplex(graph=graph)
    subsets = []
    for i in range(n_clusters):
        subset = []
        for edge in graph.edges:
            if (
                int(graph.nodes[edge[0]]["community"]) == i
                and int(graph.nodes[edge[1]]["community"]) == i
            ):
                subset.append(True)
            else:
                subset.append(False)
        subsets.append(subset)

    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    print(np.shape(harm_subspace))
    plt.figure()
    pos =nx.spring_layout(graph)
    for i, harm in enumerate(harm_subspace.T):
        plt.figure()
        nx.draw(graph, pos=pos, edge_color=abs(harm))
        c = nx.draw_networkx_edges(graph, pos=pos, edge_color=abs(harm))
        plt.colorbar(c)
        plt.savefig(f"graph_{i}.pdf", bbox_inches='tight')

    t_max = 2000
    n_t = 500
    n_min = 0

    alpha_1 = harm_subspace.sum(1) #[:, 0]
    initial_phase = np.random.random(Gsc.n_edges)

    plt.figure(figsize=(10, 4))
    for alpha_2 in [1.50]:
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

        global_order = compute_simplicial_order_parameter(result, harm_subspace)
        for i, subset in enumerate(subsets):
            order = compute_simplicial_order_parameter(result, harm_subspace, subset)
            plt.plot(time, order, label=f"subset {i}", lw=0.5)

        partial_orders = compute_harmonic_projections(result, harm_subspace)
        #plt.plot(time, global_order, label=f"alpha_2 = {alpha_2}", c='r')
        for partial_order in partial_orders:
            plt.plot(time, partial_order, label=f"partial, alpha_2 = {alpha_2}", ls='--')
        plt.plot(time, np.sum(partial_orders, axis=0), label=f"partial sum", c='r', ls='--')
    plt.gca().set_xlim(time[0], time[-1])
    plt.axhline(1.0, ls="--", c="k")
    plt.legend(loc="best")
    # plt.gca().set_ylim(0, 1.02)
    plt.xlabel("time")
    plt.ylabel("order parameter")
    plt.legend(loc="best")
    plt.savefig(f"scan_order.pdf", bbox_inches="tight")