"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def node_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None):
    """Node simplicial kuramoto, or classical Kuramoto."""
    B0 = simplicial_complex.node_incidence_matrix
    W0 = simplicial_complex.node_weights_matrix
    W1 = simplicial_complex.edge_weights_matrix

    if omega_0 is None:
        omega_0 = np.zeros(simplicial_complex.n_nodes)

    W1_inv = W1.copy()
    W1_inv.data = 1./ W1_inv.data
    return omega_0 - W0.dot(B0.T).dot(np.sin(W1_inv.dot(B0).dot(phase)))


def integrate_node_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, omega_0=None
):
    """Integrate the node Kuramoto model."""
    rhs = partial(
        node_simplicial_kuramoto, simplicial_complex=simplicial_complex, omega_0=omega_0
    )
    t_eval = np.linspace(0, t_max, n_t)
    opt = solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )    
    
    opt.y[opt.y<0] += 2*np.pi
    opt.y[opt.y>2*np.pi] -= 2*np.pi
    
    return opt


def edge_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None):
    """Edge simplicial kuramoto"""
    B0 = simplicial_complex.node_incidence_matrix
    W0 = simplicial_complex.node_weights_matrix
    B1 = simplicial_complex.edge_incidence_matrix
    W1 = simplicial_complex.edge_weights_matrix
    W2 = simplicial_complex.face_weights_matrix

    if omega_0 is None:
        omega_0 = np.zeros(simplicial_complex.n_edges)

    W1_inv = W1.copy()
    W1_inv.data = 1./ W1_inv.data
    rhs = B0.dot(W0).dot(np.sin(B0.T.dot(W1_inv).dot(phase)))

    if W2 is not None:
        W2_inv = W2.copy()
        W2_inv.data = 1./ W2_inv.data
        rhs += W1.dot(B1.T).dot(np.sin(W2_inv.dot(B1).dot(phase)))
    return omega_0 - rhs


def integrate_edge_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, omega_0=None
):
    """Integrate the edge Kuramoto model."""

    rhs = partial(
        edge_simplicial_kuramoto, simplicial_complex=simplicial_complex, omega_0=omega_0
    )
    t_eval = np.linspace(0, t_max, n_t)
    opt =  solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    opt.y[opt.y<0] += 2*np.pi
    opt.y[opt.y>2*np.pi] -= 2*np.pi
    
    return opt
