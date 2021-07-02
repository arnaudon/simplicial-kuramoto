import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import get_projection_slope, get_subspaces, proj_subspace


def get_projection_slope_custom(
    Gsc, res, grad_subspace=None, curl_subspace=None, harm_subspace=None, n_min=0
):
    """Project result on subspaces."""
    if grad_subspace is None:
        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    time = res.t[n_min:]

    grad = proj_subspace(res.y.T, grad_subspace)[n_min:]
    grad_fit = np.polyfit(time, grad, 1)
    # grad -= grad_fit[0] * time + grad_fit[1]
    # grad -= np.min(grad)

    curl = proj_subspace(res.y.T, curl_subspace)[n_min:]
    curl_fit = np.polyfit(time, curl, 1)
    # curl -= curl_fit[0] * time + curl_fit[1]
    # curl -= np.min(curl)

    harm = proj_subspace(res.y.T, harm_subspace)[n_min:]
    harm_fit = np.polyfit(time, harm, 1)
    # harm -= harm_fit[0] * time + harm_fit[1]
    # harm -= np.min(harm)

    return grad, curl, harm, grad_fit, curl_fit, harm_fit


def plot_phase_traj(Gsc, alpha_1, alpha_2, folder="figures_traj", t_max=50, min_s=1.0):

    np.random.seed(42)
    initial_phase = np.random.random(Gsc.n_edges)

    n_t = 1000
    n_min = 100

    res = integrate_edge_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
    )
    result = res.y[:, n_min:]
    time = res.t[n_min:]

    grad, curl, harm, grad_slope, curl_slope, harm_slope = get_projection_slope(
        Gsc, res, n_min=n_min
    )

    print(f"grad slope = {grad_slope}, curl slope = {curl_slope}")

    plt.figure(figsize=(3, 3))

    plt.plot(np.sin(result[0]), np.sin(result[2]), "-k")  # , c=grad, s=10 * curl + min_s)
    plt.plot(
        np.sin(result[0, ::15]), np.sin(result[2, ::15]), "k."
    )  # , c=grad, s=10 * curl + min_s)
    # plt.colorbar(label="gradient slope")
    plt.axis([-1.01, 1.01, -1.01, 1.01])
    plt.axis("equal")
    plt.suptitle(f"alpha_1 = {alpha_1}, alpha2 = {alpha_2}")
    plt.savefig(
        f"{folder}/traj_a1_{np.round(alpha_1, 5)}_a2_{np.round(alpha_2, 3)}.pdf",
        bbox_inches="tight",
    )
    plt.close()

    grad, curl, harm, grad_slope, curl_slope, harm_slope = get_projection_slope_custom(
        Gsc, res, n_min=0
    )
    time = res.t
    plt.figure(figsize=(4, 3))
    plt.plot(time, grad, label="grad")
    plt.plot(time, grad_slope[1] + time * grad_slope[0], c="k", ls="--")
    plt.plot(time, curl, label="curl")
    plt.plot(time, curl_slope[1] + time * curl_slope[0], c="k", ls="--")
    plt.xlabel("time")
    plt.ylabel("projection")
    plt.gca().set_xlim(time[0], time[-1])
    plt.legend(loc="best")
    plt.suptitle(f"alpha_1 = {alpha_1}, alpha2 = {alpha_2}")
    plt.savefig(
        f"{folder}/proj_a1_{np.round(alpha_1, 5)}_a2_{np.round(alpha_2, 3)}.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_2_phase_traj(Gsc, alphas_1, alpha_2, folder="figures_traj", t_maxs=50, min_s=1.0):

    np.random.seed(42)
    initial_phase = np.random.random(Gsc.n_edges)

    n_t = 1000
    n_min = 100

    plt.figure(figsize=(4, 3))
    results = []
    for c, alpha_1, t_max in zip(["C0", "C1"], alphas_1, t_maxs):
        res = integrate_edge_kuramoto(
            Gsc,
            initial_phase,
            t_max,
            n_t,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
        )
        results.append(res.y[:, n_min:])
        time = res.t[n_min:]

        grad, curl, harm, grad_slope, curl_slope, harm_slope = get_projection_slope_custom(
            Gsc, res, n_min=n_min
        )

        plt.plot(time, grad, label="grad", c=c, ls="--")
        plt.plot(time, curl, label="curl", c=c)

    plt.xlabel("time")
    plt.ylabel("projection")
    plt.gca().set_xlim(time[0], time[-1])
    plt.legend(loc="best")
    plt.suptitle(f"alpha_1 = {alpha_1}, alpha2 = {alpha_2}")
    plt.savefig(f"{folder}/proj_transition.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(3, 3))
    for c, result in zip(["C0", "C1"], results):
        if c =='C0':
            plt.plot(np.sin(result[0]), np.sin(result[2]), "-", c=c)
        else:
            plt.plot(np.sin(result[0]), np.sin(result[2]), "--", c=c)
        plt.plot(np.sin(result[0, ::15]), np.sin(result[2, ::15]), ".", c=c)

    plt.axis([-1.01, 1.01, -1.01, 1.01])
    plt.axis("equal")
    # plt.suptitle(f"alpha_1 = {alpha_1}, alpha2 = {alpha_2}")
    plt.savefig(f"{folder}/traj_transition.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    G = nx.Graph()
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(2, 0, weight=1, edge_com=0)

    Gsc = SimplicialComplex(graph=G, faces=[[1, 0, 2]])
    Gsc.flip_edge_orientation([0, 1])

    alphas_1 = [1.1939, 1.19395]
    t_maxs = [34.5, 34.5]
    alpha_2 = 1.5
    plot_2_phase_traj(Gsc, alphas_1, alpha_2, t_maxs=t_maxs, min_s=5, folder="figures_traj_scan")

    alpha_1 = 1.1  # 1.1939298
    alpha_2 = 1.5
    plot_phase_traj(Gsc, alpha_1, alpha_2, t_max=60)

    alpha_1 = 1.2  # 1.1939298
    alpha_2 = 1.5
    plot_phase_traj(Gsc, alpha_1, alpha_2, t_max=40)

    alpha_1 = 1.2
    alpha_2 = 2.5
    plot_phase_traj(Gsc, alpha_1, alpha_2, t_max=18, min_s=5)

