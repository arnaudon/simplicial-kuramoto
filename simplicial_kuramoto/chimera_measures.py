"""Various measure of synchronisation for chimera detection."""
import numpy as np
from itertools import combinations


def module_order_parameter(theta, community_assignment):

    Nc = len(np.unique(community_assignment))
    Nn = theta.shape[0]
    Nt = theta.shape[1]

    order_parameters = np.zeros((Nc, Nt))
    for c in range(Nc):
        comm = np.unique(community_assignment)[c]
        ind = np.argwhere(community_assignment == comm)
        order_parameters[c] = np.absolute(np.exp(1j * theta[ind]).mean(axis=0))

    global_order_parameter = np.absolute(np.exp(1j * theta).mean(axis=0))

    return order_parameters, global_order_parameter


def module_gradient_parameter(theta, community_assignment):

    phase_gradient = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        phase_gradient[i, :] = np.gradient(theta[i, :])

    Nc = len(np.unique(community_assignment))
    Nt = phase_gradient.shape[1]

    op = np.zeros((Nc, Nt))

    for c in range(Nc):
        comm = np.unique(community_assignment)[c]
        ind = np.argwhere(community_assignment == comm)
        op[c, :] = np.var(phase_gradient[ind, :], axis=0)

    return op, phase_gradient


def coalition_entropy(order_parameters, gamma=0.8):

    coalitions = (order_parameters[:-1] > gamma).T * 1
    unique_coalitions = np.unique(coalitions, axis=0)
    Nt = order_parameters.shape[1]
    M = order_parameters.shape[0] - 1

    coalition_prob = np.zeros(unique_coalitions.shape[0])

    for i in range(unique_coalitions.shape[0]):
        coalition = unique_coalitions[i, :]
        coalition_prob[i] = (coalitions == coalition).all(-1).sum() / Nt

    ce = -(coalition_prob * np.log2(coalition_prob)).sum() / M

    return ce


def pairwise_synchronisation(theta, community_assignment):

    comms = np.unique(community_assignment)
    Nt = theta.shape[1]
    Nc = len(np.unique(community_assignment))
    op = np.zeros((Nc, Nt))

    cnt = 0
    for c1, c2 in [comb for comb in combinations(comms, 2)]:
        ind1 = np.argwhere(community_assignment == c1)
        ind2 = np.argwhere(community_assignment == c2)

        op[cnt, :] = np.absolute(
            0.5 * (np.exp(1j * theta[ind1, :]).sum(0) + np.exp(1j * theta[ind2, :]).sum(0))
        ) / (len(ind1) + len(ind2))
        cnt = cnt + 1

    return op


def shanahan_indices(order_parameters):
    """
    Compute the two Shanahan indices.

    Returns:
        l is the average across communities of the variance of the order parameter
            within communities ("global" metastability)
        chi is the avarage across time of the variance of the order parameter across communities
        at time t (Chimeraness of the system)
    """
    lamb = np.var(order_parameters, axis=1).mean()
    chi = np.var(order_parameters, axis=0).mean()

    return lamb, chi
