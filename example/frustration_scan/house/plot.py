from simplicial_kuramoto.frustration_scan import (
    plot_projections,
    plot_harmonic_order,
    get_subspaces,
    compute_simplicial_order_parameter,
    compute_harmonic_projections
)
import matplotlib.pyplot as plt
from scan import my_house
import networkx as nx

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    sizes = [3]  # , 4, 5, 6, 7, 8, 9, 10]
    for size in sizes:
        Gsc = my_house(size)

        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
        path = folder + f"house_{size}.pkl"
        filename = figure_folder + f"house_{size}_proj.pdf"

        """
        plt.figure()
        graph = Gsc.graph
        harm = harm_subspace[:, 0]
        print(harm)
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos=pos)
        c = nx.draw_networkx_edges(graph, pos=pos, edge_color=abs(harm))
        plt.colorbar(c)
        plt.savefig(figure_folder + f"house_{size}_graph.pdf", bbox_inches="tight")
        plt.close()

        filename = figure_folder + f"house_{size}_flip_proj.pdf"
        plot_projections(path, filename, n_workers=80)
        plt.close()

        import pickle
        import itertools
        import numpy as np

        Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
        pairs = list(itertools.product(range(len(alpha1)), range(len(alpha2))))
        for (a1, a2), result in zip(pairs, results):
            #if abs(alpha1[a1] - 1.0) < 1e-3:
            #    if abs(alpha2[a2] - 1.3) < 1e-2:
            if abs(alpha1[a1] - 1.02) < 1e-2:
                print(alpha2[a2], alpha1[a1])
                if abs(alpha2[a2] - 1.808) < 1e-2:
                    res = result[0]
        harm_order = compute_simplicial_order_parameter(res.y, harm_subspace)
        partial_orders = compute_harmonic_projections(res.y, harm_subspace)
        print(harm_order[-10:], partial_orders[0][-10:])
        plt.figure()
        plt.plot(np.sin(res.y.T[-100:]))
        plt.plot(harm_order[-100:], c='r')
        plt.plot(partial_orders[0][-100:], c='k')
        plt.savefig("test.pdf")
        plt.close()

        """
        #plot_projections(path, filename, n_workers=80)
        filename = figure_folder + f"house_{size}_order.pdf"
        plot_harmonic_order(path, filename, n_workers=80, frac=0.5)
        #plt.close()

        """
        path = folder + f"house_{size}_harm.pkl"
        filename = figure_folder + f"house_{size}_harm_proj.pdf"
        plot_projections(path, filename, n_workers=80)
        filename = figure_folder + f"house_{size}_harm_order.pdf"
        plot_harmonic_order(path, filename, n_workers=80, frac=0.95)
        plt.close()
        """
