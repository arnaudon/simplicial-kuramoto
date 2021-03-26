"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def node_simplicial_kuramoto(time, phase, simplicial_complex=None, alpha_0=0, alpha_1=0):
    """Node simplicial kuramoto, or classical Kuramoto."""
    return -alpha_0 - simplicial_complex.lifted_N0sp.dot(
        np.sin(simplicial_complex.lifted_N0.dot(phase) + alpha_1)
    )


def integrate_node_kuramoto(simplicial_complex, initial_phase, t_max, n_t, alpha_0=0, alpha_1=0):
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


def edge_simplicial_kuramoto(time, phase, simplicial_complex=None, alpha_1=0, alpha_2=0):
    """Edge simplicial kuramoto"""
    rhs = alpha_1 + simplicial_complex.lifted_N0.dot(
        np.sin(simplicial_complex.lifted_N0sp.dot(phase))
    )
    if simplicial_complex.W2 is not None:
        rhs += simplicial_complex.lifted_N1sp.dot(
            np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_2)
        )

    return -rhs


def integrate_edge_kuramoto(simplicial_complex, initial_phase, t_max, n_t, alpha_1=0, alpha_2=0):
    """Integrate the edge Kuramoto model."""

    rhs = partial(
        edge_simplicial_kuramoto,
        simplicial_complex=simplicial_complex,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
    )
    t_eval = np.linspace(0, t_max, n_t)
    return solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def edge_simplicial_kuramoto_frustrated(
    time, phase, simplicial_complex=None, alpha_0=0, alpha_1=0, alpha_2=0
):
    B0 = simplicial_complex.B0
    W0 = simplicial_complex.W0
    B1 = simplicial_complex.B1
    W1 = simplicial_complex.W1
    W2 = simplicial_complex.W2
    LB0 = simplicial_complex.lifted_B0
    LB1 = simplicial_complex.lifted_B1
    LB0p = simplicial_complex.lifted_B0_p
    LB1p = simplicial_complex.lifted_B1_p

    Nn = B0.shape[1]
    Ne = B0.shape[0]
    Nf = B1.shape[0]

    if alpha_0 is None:
        alpha_0 = np.zeros(2 * simplicial_complex.n_edges)

    if alpha_1 is None:
        alpha_1 = np.zeros(2 * simplicial_complex.n_edges)
    else:
        alpha_1_v = np.ones(2 * Ne) * alpha_1

    if alpha_2 is None:
        alpha_2 = np.zeros(2 * simplicial_complex.n_edges)
    else:
        alpha_2_v = np.ones(Nf) * alpha_2

    rhs = -LB0.dot(np.sin(LB0p.T.dot(phase))) - alpha_1_v

    if W2 is not None:
        rhs += -LB1p.T.dot(np.sin(LB1.dot(phase) + alpha_2_v))

    return alpha_0 - rhs


def integrate_edge_kuramoto_frustrated(
    simplicial_complex, initial_phase, t_max, n_t, alpha_1=None, alpha_2=None, alpha_0=None
):
    """Integrate the frustrated edge Kuramoto model."""

    rhs = partial(
        edge_simplicial_kuramoto_frustrated,
        simplicial_complex=simplicial_complex,
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
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
    opt.y[opt.y < 0] += 2 * np.pi
    opt.y[opt.y > 2 * np.pi] -= 2 * np.pi

    return opt
