import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import scipy as sc
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random

import pickle
from tqdm import tqdm

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.graph_generator import modular_graph
from simplicial_kuramoto.integrators import *
from simplicial_kuramoto import plotting
from simplicial_kuramoto.chimera_measures import *


def plot_measures(gms, chi, ce, ceg, marker=None):

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

    return fig


def shanahan_metrics(results, alpha1, alpha2):

    gms_matrix = pd.DataFrame()
    chi_matrix = pd.DataFrame()
    ce_matrix = pd.DataFrame()
    ceg_matrix = pd.DataFrame()

    for i, (a1, a2) in enumerate(itertools.product(alpha1, alpha2)):

        gms_ = []
        chi_ = []
        ce_ = []
        ceg_ = []

        for result in results[i]:
            op, global_order_parameter = module_order_parameter(result.y, edge_community_assignment)
            gms, chi = shanahan_indices(op)
            gop, phase_gradient = module_gradient_parameter(result.y, edge_community_assignment)

            gms_.append(gms)
            chi_.append(chi)
            ce_.append(coalition_entropy(op))
            ceg_.append(coalition_entropy(gop))

        gms_matrix.loc[a1, a2] = np.mean(gms_)
        chi_matrix.loc[a1, a2] = np.mean(chi_)
        ce_matrix.loc[a1, a2] = np.mean(ce_)
        ceg_matrix.loc[a1, a2] = np.mean(ceg_)

    return gms_matrix, chi_matrix, ce_matrix, ceg_matrix


def plot_op(result, edge_community_assignment, labels=None):

    op, global_op = module_order_parameter(result.y, edge_community_assignment)
    Nc = len(np.unique(edge_community_assignment))

    plt.figure()
    for _op in op:
        plt.plot(_op)
    plt.plot(global_op)

    if labels is None:
        plt.legend(labels=["Comm1", "Comm2", "inter comm edges", "global order"])
    else:
        plt.legend(labels=labels)

    plt.xlabel("Time")
    plt.ylabel("Order parameter")
    plt.ylim([0, 1.05])


if __name__ == "__main__":
    labels = [
        "global metastability",
        "chimeraness",
        "coalition entropy",
        "coalition entropy of gradient",
    ]

    folder = "./results/"

    Gsc, results, alpha1, alpha2 = pickle.load(open(folder + "two_comms_three_nodes.pkl", "rb"))

    parameter_indexes = (
        np.linspace(0, alpha1.shape[0] * alpha2.shape[0] - 1, alpha1.shape[0] * alpha2.shape[0])
        .reshape(alpha1.shape[0], alpha2.shape[0])
        .astype(int)
    )

    edge_community_assignment = np.array(
        list(nx.get_edge_attributes(Gsc.graph, "edge_com").values())
    )
    gms, chi, ce, ceg = shanahan_metrics(results, alpha1, alpha2)

    fig = plot_measures(gms, chi, ce, ceg)
    plt.savefig("measures.pdf")


    idx_a1 = -1
    idx_a2 = 0

    fig = plot_measures(gms, chi, ce, ceg, marker=(alpha2[idx_a2], alpha1[idx_a1]))
    plt.savefig("measures.pdf")

    plot_op(results[parameter_indexes[idx_a1, idx_a2]][0], edge_community_assignment)
    plt.savefig("order_parameter.pdf")

    fig, axs = plt.subplots(len(alpha2), len(alpha1), figsize=(25, 20))
    axs = np.flip(axs, axis=0)
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        plt.sca(axs[idx_a1, idx_a2])
        plotting.plot_edge_kuramoto(results[parameter_indexes[idx_a1, idx_a2]][0])

    fig.tight_layout()
    plt.savefig("phases.pdf")


def lkklj():

    idx_a1 = 4
    idx_a2 = 10
    fig = plot_measures(gms, chi, ce, ceg, marker=(alpha2[idx_a2], alpha1[idx_a1]))
    plot_op(results[parameter_indexes[idx_a1, idx_a2]][0], edge_community_assignment)

    idx_a1 = 15
    idx_a2 = 15
    fig = plot_measures(gms, chi, ce, ceg, marker=(alpha2[idx_a2], alpha1[idx_a1]))
    plot_op(results[parameter_indexes[idx_a1, idx_a2]][0], edge_community_assignment)

    idx_a1 = 28
    idx_a2 = 15
    fig = plot_measures(gms, chi, ce, ceg, marker=(alpha2[idx_a2], alpha1[idx_a1]))
    plot_op(results[parameter_indexes[idx_a1, idx_a2]][0], edge_community_assignment)

    idx_a1 = 15
    idx_a2 = 28
    fig = plot_measures(gms, chi, ce, ceg, marker=(alpha2[idx_a2], alpha1[idx_a1]))
    plot_op(results[parameter_indexes[idx_a1, idx_a2]][0], edge_community_assignment)
