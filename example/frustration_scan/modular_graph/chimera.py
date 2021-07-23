import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.graph_generator import modular_graph
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import get_subspaces, compute_simplicial_order_parameter


if __name__ == "__main__":
    seed = 43
    np.random.seed(seed)

    n_workers = 80
    n_clusters = 3
    d = 0.3
    intra_weight = (d + 1) / 2
    inter_weight = 1 - intra_weight
    print("inter=", inter_weight, "intra=", intra_weight)

    t_max = 1000
    n_t = 5000
    n_min = 0

    alpha_2 = 1.45

    graph = modular_graph(
        n_clusters, 5, 2, inter_weight=inter_weight, intra_weight=intra_weight, rando=False
    )
    Gsc = SimplicialComplex(graph=graph)

    subsets = []
    edge_labels = {}
    for i in range(n_clusters):
        subset = []
        for edge in graph.edges:
            if (
                int(graph.nodes[edge[0]]["community"]) == i
                and int(graph.nodes[edge[1]]["community"]) == i
            ):
                subset.append(True)
                edge_labels[edge] = i
            else:
                subset.append(False)
        subsets.append(subset)

    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    plt.figure()
    pos = nx.spring_layout(graph)
    for i, harm in enumerate(harm_subspace.T):
        plt.figure()
        nx.draw(graph, pos=pos, edge_color=abs(harm))
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)
        nx.draw_networkx_labels(graph, pos=pos)
        c = nx.draw_networkx_edges(graph, pos=pos, edge_color=abs(harm))
        plt.colorbar(c)
        plt.savefig(f"graph_{i}.pdf", bbox_inches="tight")

    alpha_1 = harm_subspace.sum(1)  # [:, 0]
    initial_phase = alpha_1
    initial_phase = np.random.random(Gsc.n_edges)

    plt.figure(figsize=(10, 4))
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

    for i, subset in enumerate(subsets):
        order = compute_simplicial_order_parameter(result, Gsc, subset)
        plt.plot(time, order, label=f"subset {i}", lw=0.5)

    global_order = compute_simplicial_order_parameter(result, Gsc)
    plt.plot(time, global_order, label=f"harm order", c="r")

    plt.gca().set_xlim(time[0], time[-1])
    plt.axhline(1.0, ls="-", c="k", lw=0.5)
    plt.axhline(0.0, ls="-", c="k", lw=0.5)
    plt.legend(loc="best")
    plt.xlabel("time")
    plt.ylabel("order parameter")
    plt.legend(loc="best")
    plt.savefig(f"scan_order.pdf", bbox_inches="tight")
