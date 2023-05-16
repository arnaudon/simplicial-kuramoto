"""Example of tower simplicial kuramoto model."""
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
    t_max = 100
    n_t = 1000

    np.random.seed(40)

    sc = make_simple(plot=True, larger=True)
    sc = xgi_to_internal(sc)
    plt.savefig("sc.pdf")

    phase_init = np.random.uniform(0, np.pi, sc.n_nodes + sc.n_edges + sc.n_faces)
    res = integrate_dirac_kuramoto(
        sc,
        phase_init,
        t_max,
        n_t,
        alpha_0=1.0,
        alpha_1=1.0,
        alpha_2=0.0,
        sigma_0=1.0,
        sigma_1=1.0,
        z=2.1,
        smooth_k=2,
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

    plt.figure()
    plt.plot(res.t, node_order, label="node")
    plt.plot(res.t, edge_order, label="edge global")
    plt.plot(res.t, face_order, label="face")
    plt.legend(loc="best")
    plt.gca().set_ylim(-1, 1)
    plt.savefig("order_parameters.pdf")
    plt.show()
