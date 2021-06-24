import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import delaunay_with_holes
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import (
    get_subspaces,
    proj_subspace,
    compute_simplicial_order_parameter,
)

def get_projection_slope(
    Gsc, res, grad_subspace=None, curl_subspace=None, harm_subspace=None, n_min=0
):
    """Project result on subspaces."""
    if grad_subspace is None:
        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    time = res.t[n_min:]

    grad = proj_subspace(res.y.T, grad_subspace)[n_min:]
    grad_fit = np.polyfit(time, grad, 1)
    grad -= grad_fit[0] * time + grad_fit[1]
    grad -= np.min(grad)

    curl = proj_subspace(res.y.T, curl_subspace)[n_min:]
    curl_fit = np.polyfit(time, curl, 1)
    curl -= curl_fit[0] * time + curl_fit[1]
    curl -= np.min(curl)

    harm = proj_subspace(res.y.T, harm_subspace)[n_min:]
    harm_fit = np.polyfit(time, harm, 1)
    harm -= harm_fit[0] * time + harm_fit[1]
    harm -= np.min(harm)

    return grad, curl, harm, grad_fit[0], curl_fit[0], harm_fit[0]


def my_house(size):

    if size < 3:
        print("I can't build a house with this ...")
        return

    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=1)
    G.add_edge(1, 2, weight=1)

    for i in range(size - 2):
        G.add_edge(2 + i, 3 + i, weight=1)

    G.add_edge(i + 3, 1, weight=1)
    if size == 3:
        Gsc = SimplicialComplex(graph=G, faces=[[0, 1, 2]])
    else:
        Gsc = SimplicialComplex(graph=G)

    Gsc.flip_edge_orientation([0, size + 1])
    return Gsc



if __name__ == "__main__":
    n_workers = 80

    np.random.seed(42)
    Gsc = my_house(3)

    t_max = 500
    n_t = 500
    n_min = 0
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    initial_phase = np.random.random(Gsc.n_edges)

    alpha_1 = 2.0
    alpha_2 = 2.5
    #alpha_1, alpha_2 = 0.6, 0.9202645146879193
    #for alpha_1 in np.linspace(0.6, 0.75, 50):
    import pickle
    Gsc, results, alpha1, alpha2 = pickle.load(open('results/house_3_harm.pkl', "rb"))
    res = integrate_edge_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_1=alpha_1,# * harm_subspace[:, 0],
        alpha_2=alpha_2,
    )
    #res = results[2729][0]
    result = res.y[:, n_min:]
    time = res.t[n_min:]
    grad, curl, harm, gf, cf, hf = get_projection_slope(Gsc, res, n_min=100)
    print(gf, cf, hf)

    plt.figure()
    plt.plot(res.t[100:], grad)
    plt.savefig('grad.pdf')

    plt.figure()
    plt.plot(res.t[100:], curl)
    plt.savefig('curl.pdf')

    plt.figure()
    plt.plot(res.t[100:], harm)
    plt.savefig('harm.pdf')

    global_order, partial_orders = compute_simplicial_order_parameter(result, harm_subspace)
    print(alpha_1, np.mean(global_order[-100:]))

    plt.figure(figsize=(5, 4))
    plt.axhline(1.0, ls="--", c="k")
    plt.plot(time, global_order, c="r", label="global")
    for i, partial_order in enumerate(partial_orders):
        plt.plot(time, partial_order, lw=0.5, label=f"direction {i}")
    plt.legend(loc="best")
    plt.gca().set_ylim(0, 1.02)
    plt.xlabel("time")
    plt.ylabel("order parameter")
    plt.savefig(f"order_projs.pdf")

    plt.figure()
    plt.plot(np.sin(result.T), lw=0.1)
    plt.savefig("result.pdf")
