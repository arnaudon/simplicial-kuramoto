import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from simplicial_kuramoto import SimplicialComplex, integrate_edge_kuramoto, plotting
from simplicial_kuramoto.graph_generator import modular_graph


graph = nx.cycle_graph(4)
# graph = modular_graph(2, 10, 2)

complex_test = SimplicialComplex(graph=graph)

B0 = complex_test.node_incidence_matrix

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

plt.show()
