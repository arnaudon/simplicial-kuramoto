"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def node_simplicial_kuramoto(
    time, phase, simplicial_complex=None, alpha_0=0, alpha_1=0
):
    """Node simplicial kuramoto, or classical Kuramoto."""
    return -alpha_0 - simplicial_complex.lifted_N0sp.dot(
        np.sin(simplicial_complex.lifted_N0.dot(phase) + alpha_1)
    )


def integrate_node_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, alpha_0=0, alpha_1=0
):
    """Integrate the node Kuramoto model."""
    return solve_ivp(
        partial(
            node_simplicial_kuramoto,
            simplicial_complex=simplicial_complex,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
        ),
        [0, t_max],
        initial_phase,
        t_eval=np.linspace(0, t_max, n_t),
        method="BDF",
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def edge_simplicial_kuramoto(
    time, phase, simplicial_complex=None, alpha_1=0, alpha_2=0
):
    """Edge simplicial kuramoto"""
    rhs = alpha_1 + simplicial_complex.N0.dot(np.sin(simplicial_complex.N0s.dot(phase)))
    if simplicial_complex.W2 is not None:
        rhs += simplicial_complex.lifted_N1sp.dot(
            np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_2)
        )
    return -rhs


def integrate_edge_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, alpha_1=0, alpha_2=0
):
    """Integrate the edge Kuramoto model."""

    rhs = partial(
        edge_simplicial_kuramoto,
        simplicial_complex=simplicial_complex,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
    )
    return solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=np.linspace(0, t_max, n_t),
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )
