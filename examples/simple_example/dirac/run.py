"""Example of tower simplicial kuramoto model."""
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from simplicial_kuramoto.plotting import plot_node_kuramoto, plot_edge_kuramoto
from simplicial_kuramoto.simplicial_complex import xgi_to_internal
from simplicial_kuramoto.integrators import integrate_dirac_kuramoto
from simplicial_kuramoto.measures import (
    compute_order_parameter,
    compute_node_order_parameter,
    compute_face_order_parameter,
)

from simplicial_kuramoto.graph_generator import make_simple

if __name__ == "__main__":
    t_max = 300
    n_t = 5000

    np.random.seed(42)

    sc = make_simple(plot=False, larger=True)
    sc = xgi_to_internal(sc)

    phase_init = np.random.uniform(0, 0.3, sc.n_nodes + sc.n_edges + sc.n_faces)
    res = integrate_dirac_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=0.0,
        alpha_1=0.0,
        alpha_2=0.0,
        sigma_0=1.0,
        sigma_1=1.0,
        sigma_2=1.0,
        z=1.0,
        smooth_k=1,
    )
    node_res = deepcopy(res)
    node_res.y = res.y[: sc.n_nodes]
    edge_res = deepcopy(res)
    edge_res.y = res.y[sc.n_nodes : sc.n_nodes + sc.n_edges]
    face_res = deepcopy(res)
    face_res.y = res.y[sc.n_nodes + sc.n_edges :]

    node_order = compute_node_order_parameter(sc, node_res.y)
    edge_order = compute_order_parameter(sc, edge_res.y)[0]
    face_order = compute_face_order_parameter(sc, face_res.y)
    print(
        node_order[-1],
        edge_order[-1],
        face_order[-1],
        node_order[-1] + edge_order[-1] + face_order[-1],
    )
    plt.figure()
    plt.plot(res.t, node_order, label="node")
    plt.plot(res.t, edge_order, label="edge global")
    plt.plot(res.t, face_order, label="face")
    plt.plot(res.t, (node_order + edge_order + face_order) / 3, label="all")
    plt.legend(loc="best")
    plt.gca().set_ylim(-1.1, 1.1)
    plt.savefig("order_parameters.pdf")

    phase_amps = np.linspace(0.02, 2.0, 200)
    n = []
    e = []
    f = []
    a = []
    for phase_amp in tqdm(phase_amps):
        phase_init = np.random.uniform(0, phase_amp, sc.n_nodes + sc.n_edges + sc.n_faces)
        res = integrate_dirac_kuramoto(
            sc,
            phase_init,
            t_max,
            n_t,
            alpha_0=0.0,
            alpha_1=0.0,
            alpha_2=0.0,
            sigma_0=1.0,
            sigma_1=1.0,
            sigma_2=1.0,
            z=1.0,
            smooth_k=1,
            disable_tqdm=True,
        )
        node_res = deepcopy(res)
        node_res.y = res.y[: sc.n_nodes]
        edge_res = deepcopy(res)
        edge_res.y = res.y[sc.n_nodes : sc.n_nodes + sc.n_edges]
        face_res = deepcopy(res)
        face_res.y = res.y[sc.n_nodes + sc.n_edges :]

        node_order = compute_node_order_parameter(sc, node_res.y)
        edge_order = compute_order_parameter(sc, edge_res.y)[0]
        face_order = compute_face_order_parameter(sc, face_res.y)
        n.append(node_order[-1])
        e.append(edge_order[-1])
        f.append(face_order[-1])
        a.append((node_order[-1] + edge_order[-1] + face_order[-1]) / 3)

    plt.figure()
    plt.plot(phase_amps, n, label="node")
    plt.plot(phase_amps, e, label="edge global")
    plt.plot(phase_amps, f, label="face")
    plt.plot(phase_amps, a, label="all")
    plt.legend(loc="best")
    plt.xlabel('initial cond distance to 0')
    plt.ylabel('order parmeters')
    plt.gca().set_ylim(-1.1, 1.1)
    plt.savefig("init_scan.pdf")
    plt.show()
