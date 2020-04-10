"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def node_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None):
    """Node simplicial kuramoto, or classical Kuramoto."""
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


def edge_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None, a=1):
    """Edge simplicial kuramoto"""
    
    ## a is the "equivalent" of degree, not the overall normalisation needs to be rethought

    B0 = simplicial_complex.node_incidence_matrix
    B1 = simplicial_complex.edge_incidence_matrix
    #need to define edge and face weights matrices
    #W1 = simplicial_complex.edge_weight_matrix
    #W2 = simplicial_complex.face_weight_matrix
    degree = simplicial_complex.degree
    
    if omega_0 is None:
        omega_0 = np.zeros(simplicial_complex.n_edges)
    
    return omega_0-(B0.dot(np.sin(B0.T.dot(phase))))-a*B1.T.dot(np.sin(B1.dot(phase)))
#     this is the line for weighted version
#     return omega_0-a*np.diag(W1).dot(B0.dot(np.sin(B0.T.dot(theta))))-a*B1.T.dot(np.diag(W2).dot(np.sin(B1.dot(theta))))


def integrate_edge_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, omega_0=None, a=1
):#, weights_e, weights_f):
    """Integrate the edge Kuramoto model."""
    
    rhs = partial(
        edge_simplicial_kuramoto, simplicial_complex=simplicial_complex, omega_0=omega_0,a=1
    )    
    t_eval = np.linspace(0, t_max, n_t)
    
    return solve_ivp(
        rhs,
        # lambda t, theta: simplicial_kuramoto_full_theta(t, theta, B0, B1, omega_0, a) // old version
        [0, t_max],
        initial_phase, 
        method='Radau', ## We use Radau instead of LSODA, I'm not sure why, but I think it was because of some convergence issue
        rtol=1.49012e-8,
        atol=1.49012e-8,
    )

#     return integ.solve_ivp(lambda t, theta: simplicial_kuramoto_full_theta(t, theta, B0, B1, a, omega_0, degree, weights_n, weights_e, weights_f), [0, t_max], theta_0, t_eval = np.linspace(0, t_max, n_t),method='Radau',rtol=1.49012e-8,atol=1.49012e-8)