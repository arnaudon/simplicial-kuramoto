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
    a = (d + 1) / 2
    b = 1 - a
    print(d, a, b)
    graph = modular_graph(n_clusters, 10, 5, inter_weight=b, intra_weight=a, rando=False)
    Gsc = SimplicialComplex(graph=graph)
    print(Gsc.faces, Gsc.n_faces)

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
    print(np.shape(harm_subspace))
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

    t_max = 100
    n_t = 1000
    n_min = 0

    alpha_1 = harm_subspace.sum(1)  # [:, 0]
    # alpha_1 = harm_subspace[:, 0]
    initial_phase = alpha_1
    initial_phase = np.random.random(Gsc.n_edges)
    # alpha_2 = 1.5  # 1.55
    alpha_2 = 1.2  # 1.55

    plt.figure(figsize=(10, 4))
    for alpha_2 in [alpha_2]:
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

        for i, subset in enumerate(subsets):
            order = compute_simplicial_order_parameter(result, Gsc, subset)
            plt.plot(time, order, label=f"subset {i}", lw=0.5)

        global_order = compute_simplicial_order_parameter(result, Gsc)
        plt.plot(time, global_order, label=f"harm order", c="r")

        # partial_orders = compute_harmonic_projections(result, harm_subspace)
        # for partial_order in partial_orders:
        #    plt.plot(time, partial_order, label=f"partial, alpha_2 = {alpha_2}", ls="--")
        # plt.plot(time, np.sum(partial_orders, axis=0), label=f"partial sum", c="r", ls="--")
    plt.gca().set_xlim(time[0], time[-1])
    plt.axhline(1.0, ls="-", c="k", lw=0.5)
    plt.axhline(0.0, ls="-", c="k", lw=0.5)
    plt.legend(loc="best")
    # plt.gca().set_ylim(0, 1.02)
    plt.xlabel("time")
    plt.ylabel("order parameter")
    plt.legend(loc="best")
    plt.savefig(f"scan_order.pdf", bbox_inches="tight")

    plt.figure()
    proj = np.zeros_like(result.T)
    mask = [] #np.ones_like(harm_subspace[:, 0], dtype=bool)
    _result = result # / np.linalg.norm(result, axis=0)[np.newaxis]
    for direction in harm_subspace.T:
        proj += np.outer(_result.T.dot(direction), direction)
        mask.append(abs(direction) > 1e-10)
    mask = np.mean(mask, axis=0)
    mask = np.array(mask, dtype=bool)
    for c, subset in zip(["C0", "C1", "C2"], subsets):
        plt.plot(1e-5+abs(result.T[100:, subset] - proj[100:,subset]), "-", c=c)
        #plt.plot(abs(result.T[100:, (~mask) * subset]), "--", c=c)
    plt.axhline(1.0, ls="--", lw=0.5, c="k")
    plt.yscale('log')
    # plt.gca().set_ylim(1e-2, 1e2)

    plt.savefig("phases.pdf")
