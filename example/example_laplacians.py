import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay

from simplicial_kuramoto import SimplicialComplex


def get_grid(n_node):

    x = np.linspace(0, 1, n_node)

    points = []
    for i in range(n_node):
        for j in range(n_node):
            points.append([x[j], x[i]])

    tri = Delaunay(points)

    edge_list = []
    for t in tri.simplices:
        edge_list.append([t[0], t[1]])
        edge_list.append([t[0], t[2]])
        edge_list.append([t[1], t[2]])

    graph = nx.Graph()
    graph.add_nodes_from(np.arange(len(points)))
    graph.add_edges_from(edge_list)
    return graph, points


if __name__ == "__main__":
    graph, points = get_grid(2)

    np.random.seed(42)
    for edge in graph.edges:
        graph[edge[0]][edge[1]]["weight"] = np.random.random()

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=points, node_size=5)
    nx.draw_networkx_edges(graph, pos=points)
    plt.savefig("graph.pdf")

    complex_test = SimplicialComplex(graph=graph, no_faces=False, face_weights=[1.0, 2.0])

    print(f"L0: {complex_test.L0.toarray()}")
    print(f"L0 lifted: {complex_test.lifted_L0.toarray()}")
    print(f"diff = {np.linalg.norm(complex_test.L0.toarray() - complex_test.lifted_L0.toarray())}")

    print(f"L1: {complex_test.L1.toarray()}")
    proj_L1 = 0.5 * complex_test.V.T.dot(complex_test.lifted_L1).dot(complex_test.V).toarray()
    print(f"L1 lifted: {proj_L1}")
    print(f"diff = {np.linalg.norm(complex_test.L1.toarray() - proj_L1)}")
