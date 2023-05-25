"""Example of fustrated simplicial kuramoto model."""
import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
from simplicial_kuramoto.simplicial_complex import use_with_xgi
from tqdm import tqdm

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.graph_generator import make_simple
from simplicial_kuramoto.simplicial_complex import xgi_to_internal
from simplicial_kuramoto.frustration_scan import proj_subspace
from simplicial_kuramoto.measures import (
    compute_order_parameter,
    compute_critical_couplings,
    compute_necessary_bounds,
    compute_sufficient_bounds,
    natural_potentials,
)


if __name__ == "__main__":
    t_max = 5
    n_t = 1000
    np.random.seed(44)

    sc = make_simple(larger=True, plot=True)
    n_edges = sum(1 if len(e) == 2 else 0 for e in sc.edges.members())
    sc = xgi_to_internal(sc)
    phase_init = np.random.uniform(0, 0.5, sc.n_edges)
    sigma_up = 1.0
    sigma_down = 1.0
    alpha_1 = np.random.uniform(0, 1, sc.n_edges)

    x = linalg.null_space(sc.N0.todense())[:, 0]
    imB = linalg.orth(sc.N0s.todense())

    beta_down, beta_up = natural_potentials(sc, alpha_1)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    rEs = []
    for s in itertools.product(range(2), repeat=sc.n_nodes):
        s = np.array(s)
        Es = []
        for amp in np.linspace(-5, 5, 5000):
            if np.max(abs(beta_down / sigma_down + amp * x)) < 1.0:
                E = (-1) ** s * np.arcsin(beta_down / sigma_down + amp * x) + s * np.pi
                # minus sign because of different convention in the kuramoto integrator
                E = -E
                proj = abs(proj_subspace(np.array([E]), imB)[0] - np.linalg.norm(E))
                if proj < 2e-5:
                    print("eq solution:", proj, E)
                    rEs.append(E)

                Es.append(E)
        Es = np.array(Es)
        ax.scatter(Es[:, 0], Es[:, 1], Es[:, 2], c=Es[:, 3], s=1.0)
    rEs = np.array(rEs)
    ax.scatter(rEs[:, 0], rEs[:, 1], rEs[:, 2], c="r", s=50)
    for node_eq in rEs[:1]:
        phase_eq = np.linalg.pinv(sc.N0s.toarray()).dot(node_eq)

        edge_res = integrate_edge_kuramoto(
            sc,
            phase_eq,
            t_max,
            n_t,
            alpha_1=alpha_1,
            sigma_down=sigma_down,
            sigma_up=sigma_up,
            disable_tqdm=True,
        )

        order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
        plt.figure()
        ys = xgi_to_internal(sc).N0s.dot(edge_res.y)
        for y in ys:
            plt.plot(edge_res.t, y)
        plt.gca().set_ylim(-1, 1)
        plt.suptitle("eq solution initial cond")

    edge_res = integrate_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        variant="non_invariant",
        alpha_1=alpha_1,
        sigma_down=sigma_down,
        sigma_up=sigma_up,
        disable_tqdm=True,
    )

    order, node_order, face_order = compute_order_parameter(sc, edge_res.y)
    plt.figure()
    ys = xgi_to_internal(sc).N0s.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, non OI")

    alpha_0 = -np.arcsin(beta_down / sigma_down)

    edge_res = integrate_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        variant="non_invariant",
        sigma_down=sigma_up,
        sigma_up=sigma_down,
        disable_tqdm=True,
    )

    plt.figure()
    ys = xgi_to_internal(sc).N0s.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, constraint to 0, non OI")

    alpha_0 = -np.arcsin(beta_down / sigma_down) + node_eq

    edge_res = integrate_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        variant="non_invariant",
        sigma_down=sigma_up,
        sigma_up=sigma_down,
        disable_tqdm=True,
    )

    plt.figure()
    ys = xgi_to_internal(sc).N0s.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, constraint to -eq solution, non OI, down")

    plt.figure()
    ys = xgi_to_internal(sc).N1.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, constraint to -eq solution, non OI, up")


    beta_down = np.linalg.pinv(sc.N0s.dot(sc.lifted_N0n_right).toarray()).dot(
        sc.N0s.toarray().dot(alpha_1)
    )
    alpha_0 = -np.arcsin(beta_down / sigma_down)

    edge_res = integrate_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        sigma_down=sigma_up,
        sigma_up=sigma_down,
        disable_tqdm=True,
    )

    plt.figure()
    ys = xgi_to_internal(sc).N0s.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, constraint to 0, OI")

    phase_eq = np.linalg.pinv(sc.N0s.toarray()).dot(node_eq)
    Lup = sigma_up * sc.lifted_N1sn.dot(np.sin(sc.lifted_N1.dot(phase_eq)))
    beta_down = np.linalg.pinv(sc.N0s.dot(sc.lifted_N0n_right).toarray()).dot(
        sc.N0s.toarray().dot(alpha_1 - Lup)
    )
    alpha_0 = -np.arcsin(beta_down / sigma_down) + np.append(node_eq, -node_eq)
    edge_res = integrate_edge_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        sigma_down=sigma_up,
        sigma_up=sigma_down,
        disable_tqdm=True,
    )

    plt.figure()
    ys = xgi_to_internal(sc).N0s.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, constraint to -eq solution, OI, down")

    plt.figure()
    ys = xgi_to_internal(sc).N1.dot(edge_res.y)
    for y in ys:
        plt.plot(edge_res.t, y)
    plt.gca().set_ylim(-1, 1)
    plt.suptitle("random initial cond, constraint to -eq solution, OI, up")


    plt.show()
