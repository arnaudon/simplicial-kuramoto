"""Numerical integratiors."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def node_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None):
    """Node simplicail kuramoto, or classical Kuramoto."""
    B0 = simplicial_complex.node_incidence_matrix
    W1 = simplicial_complex.edge_weight_matrix
    degree = simplicial_complex.degree

    if omega_0 is None:
        omega_0 = np.zeros(simplicial_complex.n_nodes)

    return omega_0 - 1.0 / degree * B0.T.dot(W1).dot(np.sin(B0.dot(phase)))


def integrate_node_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, omega_0=None
):
    """Integrate the node Kuramoto model."""
    rhs = partial(
        node_simplicial_kuramoto, simplicial_complex=simplicial_complex, omega_0=omega_0
    )
    t_eval = np.linspace(0, t_max, n_t)
    return solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="LSODA",
        rtol=1.0e-8,
        atol=1.0e-8,
    )
