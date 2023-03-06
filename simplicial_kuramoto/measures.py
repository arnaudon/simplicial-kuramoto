import numpy as np
from scipy.optimize import linprog

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


def norm(v, Winv):
    """Weighted norm."""
    return np.sqrt(v.T.dot(Winv).dot(v))


@use_with_xgi
def natural_potentials(sc, alpha_1, alpha_2):
    beta_down = np.linalg.pinv(sc.N0.toarray()).dot(alpha_1)
    beta_up = np.linalg.pinv(sc.N1s.toarray()).dot(alpha_1)
    return beta_down, beta_up


@use_with_xgi
def compute_sufficient_bounds(sc, alpha_1, alpha_2, gamma=np.pi / 2.0):
    """Compute sufficient bounds for stability."""
    beta_down, beta_up = natural_potentials(sc, alpha_1, alpha_2)
    w0inv = np.linalg.inv(sc.W0.toarray())
    w2inv = np.linalg.inv(sc.W2.toarray())
    bound_down = np.sqrt(np.max(sc.W0.toarray())) * norm(beta_down, w0inv) / np.sin(gamma)
    bound_up = np.sqrt(np.max(sc.W2.toarray())) * norm(beta_up, w2inv) / np.sin(gamma)
    return bound_down, bound_up


@use_with_xgi
def compute_necessary_bounds(sc, alpha_1, alpha_2):
    """Compute necessary bounds for no phase locking.

    Sigmas smaller than these values will be guaranteed to not phase lock.
    """
    beta_down, beta_up = natural_potentials(sc, alpha_1, alpha_2)
    w0inv = np.linalg.inv(sc.W0.toarray())
    w2inv = np.linalg.inv(sc.W2.toarray())
    sigma_down = norm(beta_down, w0inv) / np.sqrt(np.diag(w0inv).sum())
    sigma_up = norm(beta_up, w2inv) / np.sqrt(np.diag(w2inv).sum())
    return sigma_down, sigma_up


@use_with_xgi
def compute_critical_couplings(sc, alpha_1, alpha_2):
    """Compute critical couplings for phase locking."""

    beta_down, beta_up = natural_potentials(sc, alpha_1, alpha_2)

    def solve(beta):
        c = np.zeros(len(beta) + 1)
        c[0] = 1.0

        A1 = np.ones(len(beta) + 1)
        A1 = np.diag(A1)
        A1[0, 0] = 0
        A1[:, 0] = -1

        A2 = -np.ones(len(beta) + 1)
        A2 = np.diag(A2)
        A2[0, 0] = 0
        A2[:, 0] = -1

        b = np.concatenate([[0], -beta])

        A = np.concatenate([A1, A2])
        b = np.concatenate([b, b])

        return linprog(c, A_ub=A, b_ub=b).fun

    sigma_down = solve(beta_down)
    sigma_up = solve(beta_up)
    return sigma_down, sigma_up
