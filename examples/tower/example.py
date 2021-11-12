import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.integrators import integrate_tower_kuramoto

if __name__ == "__main__":

    G = nx.Graph()
    G.add_edge(0, 1, weight=1, edge_com=0)
    G.add_edge(1, 2, weight=1, edge_com=0)
    G.add_edge(2, 0, weight=1, edge_com=0)

    Gsc = SimplicialComplex(graph=G)
    Gsc.flip_edge_orientation([0, 1])

    alpha1 = np.linspace(0, 2.5, 100)
    alpha2 = np.linspace(0, np.pi / 2.0, 200)
    n_repeats = 1
    t_max = 400
    n_t = 1000
    n_workers = 80
    initial_phase = np.random.normal(0, 1, Gsc.n_nodes + Gsc.n_edges)
    res = integrate_tower_kuramoto(
        Gsc,
        initial_phase,
        t_max,
        n_t,
        alpha_0=np.random.normal(0, 1, Gsc.n_nodes) * 0,
        alpha_1=2.0,
        alpha_2=1.5,
    )

    plt.figure(figsize=(10, 3))
    res.y = np.sin(res.y)
    plt.plot(res.t, res.y[0], c="k")
    plt.plot(res.t, res.y[1], c="k")
    plt.plot(res.t, res.y[2], c="k")
    plt.savefig('node.pdf')

    plt.figure(figsize=(10, 3))
    plt.plot(res.t, res.y[3], c="r")
    plt.plot(res.t, res.y[4], c="r")
    plt.plot(res.t, res.y[5], c="r")

    plt.savefig('edges.pdf')
    plt.show()
