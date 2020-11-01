import matplotlib.pyplot as plt
import numpy as np

from simplicial_kuramoto import SimplicialComplex, integrate_node_kuramoto, plotting
from simplicial_kuramoto.graph_generator import modular_graph

if __name__ == "__main__":
    graph = modular_graph(2, 10, 2)
    complex_test = SimplicialComplex(graph=graph)

    complex_test.flip_edge_orientation(0)

    initial_phase = np.zeros(len(graph))
    initial_phase[0] = 1.0

    t_max = 2.0
    n_t = 100
    results = integrate_node_kuramoto(complex_test, initial_phase, t_max, n_t)

    plotting.plot_node_kuramoto(results)
    plt.show()
