import matplotlib.pyplot as plt
import numpy as np

from simplicial_kuramoto import SimplicialComplex, integrate_node_kuramoto, plotting
from simplicial_kuramoto.graph_generator import modular_graph
import networkx as nx

if __name__ == "__main__":
    #graph = modular_graph(4, 32, 32, rando=True, inter_weight=0.5, intra_weight=0.7)
    graph = modular_graph(2, 10, 10, rando=True, inter_weight=0.7, intra_weight=1.0)
    nx.draw(graph)
    plt.savefig("graph.pdf")

    complex_test = SimplicialComplex(graph=graph, no_faces=True)

    np.random.seed(42)
    initial_phase = np.random.random(len(graph))
    #initial_phase = np.zeros(len(graph))
    #initial_phase[0] = 2.0
    #initial_phase[2] = 2.1

    t_max = 70.0
    n_t = 1000
    results = integrate_node_kuramoto(
        complex_test, initial_phase, t_max, n_t, alpha_0=0.0, alpha_1=np.pi / 2.0 - 0.1
    )

    plotting.plot_node_kuramoto(results)
    plt.savefig("example_node.pdf")
