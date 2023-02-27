"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from simplicial_kuramoto.simplicial_complex import use_with_xgi


@use_with_xgi
def compute_node_order_parameter(Gsc, result):
    """Compute the node Kuramoto order parameter."""
    w1_inv = 1.0 / np.diag(Gsc.W1.toarray())
    return w1_inv.dot(np.cos(Gsc.N0.dot(result))) / w1_inv.sum()


@use_with_xgi
def compute_order_parameter(Gsc, result, subset=None):
    """Evaluate the order parameter, or the partial one for subset edges.
    Args:
        result (array): result of simulation (edge lenght by timepoints)
        Gsc (SimplicialComplex): simplicial complex
        subset (array): bool or int array of edges in the subset to consider

    Returns:
        total order, node order, face order
    """
    w0_inv = 1.0 / np.diag(Gsc.W0.toarray())
    if Gsc.W2 is not None:
        w2_inv = 1.0 / np.diag(Gsc.W2.toarray())

    if subset is not None:
        # if we have at least an adjacent edge in subset
        w0_inv = w0_inv * np.clip(abs(Gsc.B0.T).dot(subset), 0, 1)
        # if we have all 3 edges in subset
        w2_inv = w2_inv * (abs(Gsc.B1).dot(subset) == 3)

    order_node = w0_inv.dot(np.cos(Gsc.N0s.dot(result)))
    norm_node = w0_inv.sum()

    if Gsc.W2 is not None:
        order_face = w2_inv.dot(np.cos(Gsc.N1.dot(result)))
        norm_face = w2_inv.sum()
    else:
        order_face = 0
        norm_face = 0

    return (
        (order_node + order_face) / (norm_node + norm_face),
        order_node / norm_node,
        order_face / norm_face if norm_face > 0 else 0,
    )


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


@use_with_xgi
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


@use_with_xgi
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


def adaptive_edge_simplicial_kuramoto(
    time, phase, simplicial_complex=None, alpha_1=0, alpha_2=0, sigma=1.0, pbar=None, state=None
):
    """Edge simplicial kuramoto with adaptive coupling."""
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n

    _, r_minus, r_plus = compute_order_parameter(simplicial_complex, np.array([phase]).T)
    rhs = alpha_1 + sigma * r_minus * simplicial_complex.N0.dot(
        np.sin(simplicial_complex.N0s.dot(phase))
    )
    if simplicial_complex.W2 is not None:
        rhs += (
            sigma
            * r_plus
            * simplicial_complex.lifted_N1sn.dot(
                np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_2)
            )
        )
    return -rhs


@use_with_xgi
def integrate_adaptive_edge_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,
    alpha_1=0,
    alpha_2=0,
    sigma=1.0,
    disable_tqdm=False,
):
    """Integrate the edge Kuramoto model with adaptive coupling.

    This model is inspired by

    Millán, Ana P., Joaquín J. Torres, and Ginestra Bianconi. "Explosive higher-order Kuramoto
    dynamics on simplicial complexes." Physical Review Letters 124.21 (2020): 218301.

    but with the order parameters defined via boundary operators, instead of classic ones.
    """
    with tqdm(total=n_t, disable=disable_tqdm) as pbar:
        rhs = partial(
            adaptive_edge_simplicial_kuramoto,
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
