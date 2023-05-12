"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from simplicial_kuramoto.measures import compute_order_parameter
from simplicial_kuramoto.simplicial_complex import use_with_xgi


def _update_bar(pbar, state, time):
    if pbar is not None:
        last_t, dt = state
        n = int((time - last_t) / dt)
        pbar.update(n)
        state[0] = last_t + dt * n


def node_simplicial_kuramoto(
    time, phase, simplicial_complex=None, alpha_0=0.0, alpha_1=0.0, sigma=1.0, pbar=None, state=None
):
    """Node simplicial kuramoto, or classical Kuramoto."""
    _update_bar(pbar, state, time)
    return -alpha_0 - sigma * simplicial_complex.lifted_N0sn.dot(
        np.sin(simplicial_complex.lifted_N0.dot(phase) + alpha_1)
    )


@use_with_xgi
def integrate_node_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,
    alpha_0=0.0,
    alpha_1=0.0,
    sigma=1.0,
    disable_tqdm=False,
):
    """Integrate the node Kuramoto model."""
    with tqdm(total=n_t, disable=disable_tqdm) as pbar:
        return solve_ivp(
            partial(
                node_simplicial_kuramoto,
                simplicial_complex=simplicial_complex,
                alpha_0=alpha_0,
                alpha_1=alpha_1,
                sigma=sigma,
                pbar=pbar,
                state=[0, t_max / n_t],
            ),
            [0, t_max],
            initial_phase,
            t_eval=np.linspace(0, t_max, n_t),
            method="BDF",
            rtol=1.0e-8,
            atol=1.0e-8,
        )


def edge_simplicial_kuramoto(
    time,
    phase,
    simplicial_complex=None,
    alpha_0=0.0,
    alpha_1=0.0,
    alpha_2=0.0,
    sigma_up=1.0,
    sigma_down=1.0,
    variant=None,
    variant_params=None,
    pbar=None,
    state=None,
):
    """Edge simplicial kuramoto"""
    _update_bar(pbar, state, time)

    if variant == "nonlinear":
        r, r_minus, r_plus = compute_order_parameter(simplicial_complex, np.array([phase]).T)
        rhs_minus = sigma_down * simplicial_complex.N0.dot(
            np.sin(simplicial_complex.N0s.dot(phase))
        )
        coupling_function = variant_params.get("coupling_function", "cross")
        epsilon = variant_params.get("epsilon", 1)
        if coupling_function == "cross":
            rhs = alpha_1 + (1.0 + epsilon * (r_plus - 1)) * rhs_minus
        if coupling_function == "quadratic":
            rhs = alpha_1 + (1.0 + epsilon * (r - 1)) * rhs_minus

        if simplicial_complex.W2 is not None:
            rhs_plus = sigma_up * simplicial_complex.lifted_N1sn.dot(
                np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_2)
            )
            if coupling_function == "cross":
                rhs += (1.0 + epsilon * (r_minus - 1)) * rhs_plus
            if coupling_function == "quadratic":
                rhs += (1.0 + epsilon(r - 1)) * rhs_plus
        return -rhs

    if variant == "non_invariant":
        rhs = alpha_1 + sigma_down * simplicial_complex.N0.dot(
            np.sin(simplicial_complex.N0s.dot(phase) + alpha_0)
        )
    else:

        if not isinstance(alpha_0, float) and len(alpha_0) == simplicial_complex.n_nodes:
            alpha_0 = np.append(alpha_0, alpha_0)

        rhs = alpha_1 + sigma_down * simplicial_complex.lifted_N0n.dot(
            np.sin(simplicial_complex.lifted_N0s.dot(phase) + alpha_0)
        )

    if simplicial_complex.W2 is not None:
        if not isinstance(alpha_2, float) and len(alpha_2) == simplicial_complex.n_faces:
            alpha_2 = np.append(alpha_2, alpha_2)

        if variant == "non_invariant":
            rhs += sigma_up * simplicial_complex.N1s.dot(
                np.sin(simplicial_complex.N1.dot(phase) + alpha_2)
            )
        else:
            rhs += sigma_up * simplicial_complex.lifted_N1sn.dot(
                np.sin(simplicial_complex.lifted_N1.dot(phase) + alpha_2)
            )
    return -rhs


@use_with_xgi
def integrate_edge_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,
    alpha_0=0.0,
    alpha_1=0.0,
    alpha_2=0.0,
    sigma_down=1.0,
    sigma_up=1.0,
    disable_tqdm=False,
    variant=None,
    variant_params=None,
):
    """Integrate the edge Kuramoto model.

    Several variants are implemented from this function.

    By default, the model is the Kuramoto-Sakaguchi with orientation invariant frustration.

    If variant='non_invariant' the orientation invariance is dropped.

    If variant='nonlinear', the nonlinear, or explosive version is used.
    This variant has the following parameters:
        - coupling_function:  cross or quadratic
        - epsilon: parameter for the coupling function

    Args:
        simplicial_complex (either xgi or internal): simplicial complex to support the dynamics
        initial_phase (vector): initial edge phases
        t_max (float): max time integration
        n_t (int): number of timesteps
        alpha_1 (float/vector): natural frequency/ies (len(edges) size)
        alpha_2 (float/vector): face frustration/s (len(faces) size)
        sigma_down (float/vector): down term amplitude/s (len(edges) size)
        sigma_up (float/vector): up term amplitude/s (len(edges) size)
        disable_tqdm (bool): show progress bar or not
        variant (str): can be None or 'non_invariant' or 'nonlinear'
        variant_params (dict): params for the variant (see docstring)
    """
    with tqdm(total=n_t, disable=disable_tqdm) as pbar:
        rhs = partial(
            edge_simplicial_kuramoto,
            simplicial_complex=simplicial_complex,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_up=sigma_up,
            sigma_down=sigma_down,
            variant=variant,
            variant_params=variant_params,
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
    alpha_0=None,
    alpha_1=None,
    alpha_2=None,
    sigma_0=1.0,
    sigma_1=1.0,
    pbar=None,
    state=None,
):
    """Tower kuramoto."""
    _update_bar(pbar, state, time)
    phase_node = phase[: simplicial_complex.n_nodes]
    phase_edge = phase[simplicial_complex.n_nodes :]

    sol_node = node_simplicial_kuramoto(
        time,
        phase_node,
        simplicial_complex=simplicial_complex,
        alpha_0=alpha_0 * simplicial_complex.N0s.dot(phase_edge),
        alpha_1=phase_edge,
        sigma=sigma_0,
    )
    sol_edge = edge_simplicial_kuramoto(
        time,
        phase_edge,
        simplicial_complex=simplicial_complex,
        alpha_1=alpha_1 * simplicial_complex.N0.dot(phase_node),
        alpha_2=alpha_2,
        sigma_up=sigma_1,
        sigma_down=sigma_1,
        pbar=pbar,
        state=state,
    )
    return np.append(sol_node, sol_edge)


@use_with_xgi
def integrate_tower_kuramoto(
    simplicial_complex,
    initial_phase,
    t_max,
    n_t,
    alpha_0=0,
    alpha_1=0,
    alpha_2=0,
    sigma_0=1.0,
    sigma_1=1.0,
    disable_tqdm=False,
):
    """Integrate the tower Kuramoto model."""
    with tqdm(total=n_t, disable=disable_tqdm) as pbar:
        rhs = partial(
            tower_kuramoto,
            simplicial_complex=simplicial_complex,
            alpha_0=alpha_0,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
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
