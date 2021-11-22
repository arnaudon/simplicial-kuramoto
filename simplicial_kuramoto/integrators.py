"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def node_simplicial_kuramoto(
    time, phase, simplicial_complex=None, alpha_0=0, alpha_1=0, sigma=1.0, pbar=None, state=None
):
    """Node simplicial kuramoto, or classical Kuramoto."""
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n

    if not isinstance(alpha_1, float):
        alpha_1 = np.append(alpha_1, alpha_1)

    return -alpha_0 - sigma * simplicial_complex.lifted_N0sn.dot(
        np.sin(simplicial_complex.lifted_N0.dot(phase) + alpha_1)
    )


def integrate_node_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, alpha_0=0, alpha_1=0, sigma=1.0
):
    """Integrate the node Kuramoto model."""
    return solve_ivp(
        partial(
            node_simplicial_kuramoto,
            simplicial_complex=simplicial_complex,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            sigma=sigma,
        ),
        [0, t_max],
        initial_phase,
        t_eval=np.linspace(0, t_max, n_t),
        method="BDF",
        rtol=1.0e-8,
        atol=1.0e-8,
    )


def edge_simplicial_kuramoto(
    time, phase, simplicial_complex=None, alpha_1=0, alpha_2=0, sigma=1.0, pbar=None, state=None
):
    """Edge simplicial kuramoto"""
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n

    rhs = alpha_1 + sigma * simplicial_complex.N0.dot(np.sin(simplicial_complex.N0s.dot(phase)))
    if simplicial_complex.W2 is not None:
        rhs += sigma * simplicial_complex.lifted_N1sn.dot(
            np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_2)
        )
    return -rhs


def integrate_edge_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,
    alpha_1=0,
    alpha_2=0,
    sigma=1.0,
    disable_tqdm=False,
):
    """Integrate the edge Kuramoto model."""

    with tqdm(total=n_t, disable=disable_tqdm) as pbar:

        rhs = partial(
            edge_simplicial_kuramoto,
            simplicial_complex=simplicial_complex,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma=sigma,
            pbar=pbar,
            state=[0, t_max / n_t],
        )
        return solve_ivp(
            rhs,
            [0, t_max],
            initial_phase,
            t_eval=np.linspace(0, t_max, n_t),
            method="BDF",
            rtol=1.0e-8,
            atol=1.0e-8,
        )
