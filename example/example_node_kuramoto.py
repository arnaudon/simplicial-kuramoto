import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex, integrate_node_kuramoto, plotting
from simplicial_kuramoto.graph_generator import modular_graph


graph = modular_graph(2, 10, 2)
complex_test = SimplicialComplex(graph=graph)

print("Incidence matrix:", complex_test.node_incidence_matrix)
complex_test.flip_edge_orientation(0)
print("Orientation of 0 fliped:", complex_test.node_incidence_matrix)


initial_phase = np.zeros(len(graph))
initial_phase[0] = 1.0

results = integrate_node_kuramoto(complex_test, initial_phase, 20.0, 100)

plotting.plot_node_kuramoto(results)
plt.show()
