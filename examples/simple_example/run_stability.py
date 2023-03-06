"""Example of fustrated simplicial kuramoto model."""
import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.integrators import integrate_node_kuramoto
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.integrators import (
    compute_node_order_parameter,
    compute_order_parameter,
)
from simplicial_kuramoto.plotting import plot_node_kuramoto, plot_edge_kuramoto
from make_sc import make_sc
from simplicial_kuramoto.simplicial_complex import use_with_xgi


@use_with_xgi
def compute_stability(sc, alpha_1, alpha_2):
    beta_down = np.linalg.pinv(sc.N0.toarray()).dot(alpha_1)
    beta_up = np.linalg.pinv(sc.N1s.toarray()).dot(alpha_1)
    sigma_down = np.linalg.norm(beta_down) / np.sqrt((1 / np.diag(sc.W0.toarray())).sum())
    sigma_up = np.linalg.norm(beta_up) / np.sqrt((1 / np.diag(sc.W2.toarray())).sum())
    return sigma_down, sigma_up


if __name__ == "__main__":
    t_max = 1000
    n_t = 1000
    n_workers = 10
    repeats = 10
    n_alpha2 = 200
    np.random.seed(2)

    sc = make_sc()
    phase_init = np.random.uniform(0, np.pi, 5)
    alpha_1 = np.random.uniform(0, 1, 5)
    alpha_2 = 0.0

    sigma_down, sigma_up = compute_stability(sc, alpha_1, alpha_2)
    print(sigma_down, sigma_up)
    sigmas = np.linspace(0.05, 0.8, 100)
    o = []
    n = []
    f = []
    for sigma in sigmas:
        edge_res = integrate_edge_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma=sigma,
        )
        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        o.append(np.mean(order[-200:]))
        n.append(np.mean(node_order[-200:]))
        f.append(np.mean(face_order[-200:]))
    plt.figure()
    plt.plot(sigmas, o, label='order')
    plt.plot(sigmas, n, label='node')
    plt.plot(sigmas, f, label='face')
    plt.axvline(sigma_down, label='down', c='r')
    plt.axvline(sigma_up, label='up', c='k')
    plt.legend()
    plt.savefig('stability_test.pdf')
