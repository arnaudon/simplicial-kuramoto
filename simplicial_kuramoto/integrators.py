"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def node_simplicial_kuramoto(
    time, phase, simplicial_complex=None, natural_frequency=0, alpha_upper=0, sigma=1.0, pbar=None, state=None
):
    """Node simplicial kuramoto, or classical Kuramoto."""
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n

    if not isinstance(alpha_upper, float):
        alpha_upper = np.append(alpha_upper, alpha_upper)

    return - natural_frequency - sigma * simplicial_complex.lifted_N0.T.dot(
        np.sin(simplicial_complex.lifted_N0.dot(phase) + alpha_upper)
    )


def integrate_node_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, natural_frequency=0, alpha_upper=0, sigma=1.0
):
    """Integrate the node Kuramoto model."""
    return solve_ivp(
        partial(
            node_simplicial_kuramoto,
            simplicial_complex=simplicial_complex,
            natural_frequency=natural_frequency,
            alpha_upper=alpha_upper,
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
    time, phase, simplicial_complex=None, natural_frequency=0, alpha_lower=0, alpha_upper=0, sigma_lower=1.0, sigma_upper=1.0, pbar=None, state=None
):
    """Edge simplicial kuramoto"""
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n

   # if not isinstance(alpha_lower, float) and not isinstance(alpha_lower, int):
        #alpha_lower = np.append(alpha_lower, alpha_lower)

    if not (isinstance(alpha_lower, float) or isinstance(alpha_lower, int)):
        if alpha_lower.shape[0]==simplicial_complex.n_nodes:
            alpha_lower = np.append(alpha_lower, alpha_lower)

    #rhs = natural_frequency + sigma_lower * simplicial_complex.N0.dot(np.sin(simplicial_complex.N0s.dot(phase)))
    rhs = natural_frequency + sigma_lower * simplicial_complex.lifted_N0sn.dot(np.sin(simplicial_complex.lifted_N0s.dot(phase) + alpha_lower))

    if simplicial_complex.W2 is not None:
        rhs += sigma_upper * simplicial_complex.lifted_N1sn.dot(
            np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_upper)
        )
    return -rhs


def integrate_edge_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,
    natural_frequency=0,
    alpha_lower=0,
    alpha_upper=0,
    sigma_lower=1.0,
    sigma_upper=1.0,
    disable_tqdm=False,
):
    """Integrate the edge Kuramoto model."""

    with tqdm(total=n_t, disable=disable_tqdm) as pbar:

        rhs = partial(
            edge_simplicial_kuramoto,
            simplicial_complex=simplicial_complex,
            natural_frequency=natural_frequency,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
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


def tower_kuramoto(
    time,
    phase,
    simplicial_complex=None,
    pbar=None,
    state=None,
    natural_frequency_node=None,
    natural_frequency_edge=None,
    alpha_upper_node=None,
    alpha_lower_edge=None,
    alpha_upper_edge=None,
    sigma_node_edge=1.0,
    sigma_edge_face=1.0,
):
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n

    phase_node = phase[: simplicial_complex.n_nodes]
    phase_edge = phase[simplicial_complex.n_nodes :]

    sol_node = node_simplicial_kuramoto(
        time,
        phase_node,
        simplicial_complex=simplicial_complex,
        natural_frequency=natural_frequency_node,# -alpha_0 * simplicial_complex.N0s.dot(phase_edge),
        alpha_upper=alpha_upper_node*phase_edge,
        sigma=sigma_node_edge,
        pbar=None,
        state=None,
    )
    sol_edge = edge_simplicial_kuramoto(
        time,
        phase_edge,
        simplicial_complex=simplicial_complex,    
        natural_frequency=natural_frequency_edge,
        alpha_lower=alpha_lower_edge*phase_node, # needed
        alpha_upper=alpha_upper_edge, # needed
        sigma_lower=sigma_node_edge,
        sigma_upper=sigma_edge_face,
        pbar=None,
        state=None,
    )
    return np.append(sol_node, sol_edge)


def integrate_tower_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,    
    natural_frequency_node=0,
    natural_frequency_edge=0,
    alpha_upper_node=0,
    alpha_lower_edge=0,
    alpha_upper_edge=0,
    sigma_node_edge=1.0,
    sigma_edge_face=1.0,
    disable_tqdm=False,
):
    """Integrate the edge Kuramoto model."""
    with tqdm(total=n_t, disable=disable_tqdm) as pbar:

        rhs = partial(
            tower_kuramoto,
            simplicial_complex=simplicial_complex,
            natural_frequency_node=natural_frequency_node,
            natural_frequency_edge=natural_frequency_edge,
            alpha_upper_node=alpha_upper_node,
            alpha_lower_edge=alpha_lower_edge,
            alpha_upper_edge=alpha_upper_edge,
            sigma_node_edge=sigma_node_edge,
            sigma_edge_face=sigma_node_edge,
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
