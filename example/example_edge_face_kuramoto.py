import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from simplicial_kuramoto import SimplicialComplex, integrate_edge_kuramoto, plotting

if __name__ == "__main__":
    graph = nx.cycle_graph(3)

    complex_test = SimplicialComplex(graph=graph, faces=[[0, 1, 2]])

    initial_phase = np.random.uniform(0, 2 * np.pi, len(graph.edges))
    initial_phase = np.zeros(len(graph.edges))
    initial_phase[0] = 1.0

    t_max = 10
    n_t = 100

    edge_result = integrate_edge_kuramoto(complex_test, initial_phase, t_max, n_t)
    plotting.plot_edge_kuramoto(edge_result)

    # test invariance of solution w.r.t orientation
    complex_test.flip_edge_orientation(2)
    edge_result = integrate_edge_kuramoto(complex_test, initial_phase, t_max, n_t)
    edge_result.y[2] *= -1
    plotting.plot_edge_kuramoto(edge_result)

    # test invariance face orientation
    complex_test = SimplicialComplex(graph=graph, faces=[[0, 2, 1]])

    edge_result = integrate_edge_kuramoto(complex_test, initial_phase, t_max, n_t)
    plotting.plot_edge_kuramoto(edge_result)

    # test invariance of solution w.r.t orientation
    complex_test.flip_edge_orientation(2)
    edge_result = integrate_edge_kuramoto(complex_test, initial_phase, t_max, n_t)
    edge_result.y[2] *= -1
    plotting.plot_edge_kuramoto(edge_result)

    plt.show()
