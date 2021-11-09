"""Various measure of synchronisation for chimera detection on node Kuramoto.

WARNING: not used and experimental, so be careful.
"""
import pickle
from itertools import combinations, product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def module_order_parameter(theta, community_assignment):
    """Order parmater."""
    Nc = len(np.unique(community_assignment))
    Nt = theta.shape[1]

    order_parameters = np.zeros((Nc, Nt))
    for c in range(Nc):
        comm = np.unique(community_assignment)[c]
        ind = np.argwhere(community_assignment == comm)
        order_parameters[c] = np.absolute(np.exp(1j * theta[ind]).mean(axis=0))

    global_order_parameter = np.absolute(np.exp(1j * theta).mean(axis=0))

    return order_parameters, global_order_parameter


def module_gradient_parameter(theta, community_assignment):
    "Gradient parameter." ""
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
    """Coalition entropy."""
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
    """Pairwise synchronisation."""
    comms = np.unique(community_assignment)
    Nt = theta.shape[1]
    Nc = len(np.unique(community_assignment))
    op = np.zeros((Nc, Nt))

    cnt = 0
    for c1, c2 in combinations(comms, 2):
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


def shanahan_metrics(results, alpha1, alpha2, edge_community_assignment):
    """Shanahan metrics."""
    gms_matrix = pd.DataFrame()
    chi_matrix = pd.DataFrame()
    ce_matrix = pd.DataFrame()
    ceg_matrix = pd.DataFrame()

    for i, (a1, a2) in enumerate(product(alpha1, alpha2)):

        gms_ = []
        chi_ = []
        ce_ = []
        ceg_ = []

        for result in results[i]:
            op, _ = module_order_parameter(result.y, edge_community_assignment)
            gms, chi = shanahan_indices(op)
            gop, _ = module_gradient_parameter(result.y, edge_community_assignment)

            gms_.append(gms)
            chi_.append(chi)
            ce_.append(coalition_entropy(op))
            ceg_.append(coalition_entropy(gop))

        gms_matrix.loc[a1, a2] = np.mean(gms_)
        chi_matrix.loc[a1, a2] = np.mean(chi_)
        ce_matrix.loc[a1, a2] = np.mean(ce_)
        ceg_matrix.loc[a1, a2] = np.mean(ceg_)

    return gms_matrix, chi_matrix, ce_matrix, ceg_matrix


def plot_measures(path, filename, marker=None):
    """Plot some of the above measures."""

    with open(path, "rb") as pkl:
        Gsc, results, alpha1, alpha2 = pickle.load(pkl)
    edge_community_assignment = np.array(
        list(nx.get_edge_attributes(Gsc.graph, "edge_com").values())
    )

    gms, chi, ce, ceg = shanahan_metrics(results, alpha1, alpha2, edge_community_assignment)

    fig, axs = plt.subplots(2, 2)
    extents = [gms.index[0], gms.index[-1], gms.columns[0], gms.columns[-1]]

    cm = axs[0, 0].imshow(gms.to_numpy(), aspect="auto", extent=extents, origin="lower")
    axs[0, 0].set_title("Global metastability")
    plt.colorbar(cm, ax=axs[0, 0])
    if marker is not None:
        axs[0, 0].scatter(marker[0], marker[1], s=100, c="red", marker="o")

    cm = axs[0, 1].imshow(chi.to_numpy(), aspect="auto", extent=extents, origin="lower")
    axs[0, 1].set_title("Chimeraness")
    plt.colorbar(cm, ax=axs[0, 1])
    if marker is not None:
        axs[0, 1].scatter(marker[0], marker[1], s=100, c="red", marker="o")

    cm = axs[1, 0].imshow(ce.to_numpy(), aspect="auto", extent=extents, origin="lower")
    axs[1, 0].set_title("Coalition Entropy")
    plt.colorbar(cm, ax=axs[1, 0])
    if marker is not None:
        axs[1, 0].scatter(marker[0], marker[1], s=100, c="red", marker="o")

    cm = axs[1, 1].imshow(ceg.to_numpy(), aspect="auto", extent=extents, origin="lower")
    axs[1, 1].set_title("Gradient coalition entropy")
    plt.colorbar(cm, ax=axs[1, 1])
    if marker is not None:
        axs[1, 1].scatter(marker[0], marker[1], s=100, c="red", marker="o")

    fig.tight_layout()

    fig.text(0.5, 0.0, "Alpha 2", ha="center")
    fig.text(0.0, 0.5, "Alpha 1", va="center", rotation="vertical")

    plt.savefig(filename, bbox_inches="tight")
