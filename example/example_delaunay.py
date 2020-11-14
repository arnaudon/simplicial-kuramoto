import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import pylab as plt

from simplicial_kuramoto import SimplicialComplex, integrate_edge_kuramoto, plotting

np.random.seed(0)


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
    graph, points = get_grid(3)
    edge_flip = 5

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=points, node_size=5)
    nx.draw_networkx_edges(graph, pos=points)
    nx.draw_networkx_edges(
        graph,
        edgelist=[list(graph.edges())[edge_flip]],
        pos=points,
        edge_color="r",
        width=2,
    )
    edge_labels = dict(((u, v), d) for d, (u, v) in enumerate(graph.edges))
    node_labels = dict((u, d) for d, u in enumerate(graph.nodes))
    nx.draw_networkx_edge_labels(graph, pos=points, edge_labels=edge_labels)
    nx.draw_networkx_labels(graph, pos=points, labels=node_labels)
    plt.savefig("delaunay_graph.png")

    np.random.seed(20)
    initial_phase = -np.random.uniform(0, 2 * np.pi, len(graph.edges))
    #initial_phase = np.zeros(len(graph.edges))
    #initial_phase[edge_flip] = 1.0
    initial_phase /= np.linalg.norm(initial_phase)

    t_max = 50
    n_t = 1000

    complex_delaunay = SimplicialComplex(graph=graph, no_faces=True)

    edge_result = integrate_edge_kuramoto(complex_delaunay, initial_phase, t_max, n_t)
    L1 = complex_delaunay.edge_laplacian.toarray()
    print('dim(ker)  no face:', np.shape(L1)[0] - np.linalg.matrix_rank(L1), np.shape(L1))
    w,  v = np.linalg.eigh(L1)
    print('eigenvalues:',  np.sort(w))
    plt.figure()
    nx.draw_networkx_nodes(graph, pos=points, node_size=5)
    edge=nx.draw_networkx_edges(
        graph,
        pos=points,
        edge_color=v[:, np.argmin(w)],
        edge_cmap=plt.get_cmap("bwr"),
        width=5,
    )
    plt.colorbar(edge)
    plt.suptitle('smallest eigenvector')


    errors = []
    plt.figure()
    for t, phase in zip(edge_result.t, edge_result.y.T):
        err = L1.dot(phase)
        err = np.round(err + np.pi, 2) % (2 * np.pi)  - np.pi
        errors.append(np.mean(err))
        plt.plot(t* np.ones(len(err)), err, '.', c='0.5')
    plt.plot(edge_result.t, errors)
    plt.savefig("error_noface.png")

    plotting.plot_edge_kuramoto(edge_result)
    plt.savefig("phases_no_faces.png")

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=points, node_size=5)
    edge = nx.draw_networkx_edges(
        graph,
        pos=points,
        edge_color=np.round(edge_result.y[:, -1] + np.pi, 2) % (2 * np.pi)  - np.pi,
        edge_cmap=plt.get_cmap("bwr"),
        width=5,
        #edge_vmin=np.min(edge_result.y),
        #edge_vmax=np.max(edge_result.y),
    )
    plt.colorbar(edge)
    nx.draw_networkx_edge_labels(graph, pos=points, edge_labels=edge_labels)
    plt.suptitle('stationary state')
    plt.savefig("large_time_no_faces.png")

    complex_delaunay = SimplicialComplex(graph=graph, no_faces=False)
    #complex_delaunay = SimplicialComplex(graph=graph, faces=complex_delaunay.faces[1:7])

    edge_result = integrate_edge_kuramoto(complex_delaunay, initial_phase, t_max, n_t)
    L1 = complex_delaunay.edge_laplacian.toarray()
    print('dim(ker) faces:', np.shape(L1)[0] - np.linalg.matrix_rank(L1), np.shape(L1))

    w,  v = np.linalg.eigh(L1)
    print('eigenvalues:', np.sort(w))

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=points, node_size=5)
    edge=nx.draw_networkx_edges(
        graph,
        pos=points,
        edge_color=v[:, np.argmin(w)],
        edge_cmap=plt.get_cmap("bwr"),
        width=5,
    )
    plt.colorbar(edge)
    plt.suptitle('smallest eigenvector')

    errors = []
    plt.figure()
    for t, phase in zip(edge_result.t, edge_result.y.T):
        err = L1.dot(phase)
        err = np.round(err + np.pi, 2) % (2 * np.pi)  - np.pi
        plt.plot(t* np.ones(len(err)), err, '.', c='0.5')
        errors.append(np.mean(err))
    plt.plot(edge_result.t, errors)
    plt.savefig("error_face.png")

    plotting.plot_edge_kuramoto(edge_result)
    plt.savefig("phases_faces.png")

    plt.figure()
    nx.draw_networkx_nodes(graph, pos=points, node_size=5)
    edge = nx.draw_networkx_edges(
        graph,
        pos=points,
        edge_color=np.round(edge_result.y[:, -1] + np.pi, 2) % (2 * np.pi)  - np.pi,
        edge_cmap=plt.get_cmap("bwr"),
        width=5,
        #edge_vmin=np.min(edge_result.y),
        #edge_vmax=np.max(edge_result.y),
    )
    plt.colorbar(edge)
    nx.draw_networkx_edge_labels(graph, pos=points, edge_labels=edge_labels)
    plt.suptitle('stationary state')

    plt.savefig("large_time_faces.png")

    plt.show()
