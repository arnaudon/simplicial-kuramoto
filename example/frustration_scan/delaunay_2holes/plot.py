import pickle
import numpy as np
import matplotlib.pyplot as plt
from simplicial_kuramoto.frustration_scan import (
    plot_projections,
    plot_harmonic_order_1d,
    get_subspaces,
    compute_simplicial_order_parameter,
)


if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    radius = 0.15
    path = folder + f"delaunay_one_hole_{radius}_harmonic.pkl"

    n_workers = 80

    filename = figure_folder + "delaunay_one_hole_1d_harmonic_order.pdf"
    plot_harmonic_order_1d(path, filename, n_workers=n_workers, frac=0.9)

def lkj():
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    alpha_i =  -1
    print(alpha2[alpha_i])
    result = results[alpha_i][0]
    res = result.y

    subset = np.zeros(Gsc.n_edges, dtype=bool)
    subset[:int(Gsc.n_edges/3)] = True
    order_1 = compute_simplicial_order_parameter(res, Gsc, subset=list(subset))

    subset = np.zeros(Gsc.n_edges, dtype=bool)
    subset[int(2*Gsc.n_edges/3):] = True
    order_2 = compute_simplicial_order_parameter(res, Gsc, subset=list(subset))

    order = compute_simplicial_order_parameter(res, Gsc)
    plt.figure()
    plt.plot(result.t, order, label='order')
    plt.plot(result.t, order_1, label='order1')
    plt.plot(result.t, order_2, label='order2')
    plt.legend()
    plt.savefig('partial.pdf')
