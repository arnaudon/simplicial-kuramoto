import numpy as np
import sys
from copy import copy
import matplotlib.pyplot as plt
import networkx as nx

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters, proj_subspace
from simplicial_kuramoto.graph_generator import delaunay_with_holes
from simplicial_kuramoto.frustration_scan import get_subspaces, compute_simplicial_order_parameter

if __name__ == "__main__":

    np.random.seed(42)
    remove_n_points = 2
    n_points = 6
    face = 2
    points = [[x, y] for x in np.linspace(0, 1, 3) for y in np.linspace(0, 1, 3)]
    p_list = [8, 6, 2, 0]
    points = [p for i, p in enumerate(points) if i not in p_list[:remove_n_points]]
    graph, points = delaunay_with_holes(n_points, points=points)

    Gsc = SimplicialComplex(graph=graph)
    faces = Gsc.faces
    del faces[face]
    Gsc = SimplicialComplex(graph=graph, faces=faces)

    t_max = 100
    n_t = 1000
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    alpha_1 = 1.5*harm_subspace.sum(1)
    alpha_2 = 1.0

    initial_phase = np.random.random(Gsc.n_edges)
    result = integrate_edge_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
    )
    order, order_node, order_face = compute_simplicial_order_parameter(result.y, Gsc)
    plt.figure()
    plt.plot(order)
    plt.savefig("curl_order.pdf")

    plt.figure()
    print('curl:', np.shape(curl_subspace))
    print('grad:', np.shape(grad_subspace))
    print('harm:', np.shape(harm_subspace))
    for direction in curl_subspace.T:
        plt.plot(result.y.T.dot(direction), lw=1)
    plt.plot(proj_subspace(result.y.T, curl_subspace), c='r')
    plt.savefig("curl.pdf")

    plt.figure()
    for direction in grad_subspace.T:
        plt.plot(result.y.T.dot(direction), lw=1)
    plt.savefig("grad.pdf")

    plt.figure()
    for direction in harm_subspace.T:
        plt.plot(result.y.T.dot(direction), lw=1)
    plt.savefig("harm.pdf")
