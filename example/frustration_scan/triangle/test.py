import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.frustration_scan import get_projection_slope, get_subspaces


def plot_phase_traj(Gsc, alpha_1, alpha_2, folder="figures_traj", t_max=50, min_s=1.0):

    np.random.seed(42)
    initial_phase = np.random.random(Gsc.n_edges)

    n_t = 500
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

    plt.figure(figsize=(4, 4))

    plt.scatter(np.sin(result[0]), np.sin(result[2]), c=grad, s=10 * curl + min_s)
    # plt.colorbar(label="gradient slope")
    plt.axis([-1.01, 1.01, -1.01, 1.01])
    plt.axis("equal")
    plt.suptitle(f"alpha_1 = {alpha_1}, alpha2 = {alpha_2}")
    plt.savefig(
        f"{folder}/traj_a1_{np.round(alpha_1, 5)}_a2_{np.round(alpha_2, 3)}.pdf",
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(time, grad, label="grad")
    plt.plot(time, curl, label="curl")
    plt.legend(loc="best")
    plt.suptitle(f"alpha_1 = {alpha_1}, alpha2 = {alpha_2}")
    plt.savefig(
        f"{folder}/proj_a1_{np.round(alpha_1, 5)}_a2_{np.round(alpha_2, 3)}.pdf",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":

    G = nx.Graph()
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(2, 0, weight=1, edge_com=0)

    Gsc = SimplicialComplex(graph=G, faces=[[0, 1, 2]])
    #Gsc.flip_edge_orientation(1)

    N0 = Gsc.N0.toarray()
    N0s = Gsc.N0s.toarray()
    N1 = Gsc.N1.toarray()
    N1s = Gsc.N1s.toarray()
    V1 = Gsc.V1.toarray()
    print(Gsc.edgelist)

    print("N0", N0)
    print("N0s", N0s)
    print("N1", N1)
    print("N1s", N1s)

    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    print('grad', grad_subspace)
    print('curl', curl_subspace)
    initial_phase = np.random.random(3)
    t_max = 10
    n_t = 100
    alpha_1 = 1.10
    alpha_2 = 1.0
    res = integrate_edge_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
    )
    plt.figure()
    for _res in res.y:
        plt.plot(res.t, np.sin(_res))
    plt.savefig('test.pdf')


