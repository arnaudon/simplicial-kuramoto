"""Tools to scan frustration parameters."""
import pandas as pd
import logging
import networkx as nx
from functools import partial
import itertools
import os

import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from multiprocessing import Pool

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.chimera_measures import (
    module_order_parameter,
    module_gradient_parameter,
    shanahan_indices,
    coalition_entropy,
)


L = logging.getLogger(__name__)


def scan_frustration_parameters(
    simplicial_complex,
    alpha1=np.linspace(0, np.pi, 5),
    alpha2=np.linspace(0, np.pi, 5),
    repeats=20,
    n_workers=4,
    t_max=100,
    n_t=1000,
    save=True,
    folder="./results/",
    filename="results.pkl",
    initial_phase=None,
):

    if not os.path.exists(folder):
        os.makedirs(folder)

    parameter_combinations = list(itertools.product(alpha1, alpha2))

    results = compute_scan(
        simplicial_complex,
        parameter_combinations,
        n_workers=n_workers,
        repeats=repeats,
        t_max=t_max,
        n_t=n_t,
        initial_phase=initial_phase,
    )

    if save:
        with open(folder + filename, "wb") as f:
            pickle.dump([simplicial_complex, results, alpha1, alpha2], f)

    return results


def integrate_kuramoto(
    parameters, simplicial_complex, repeats, t_max, n_t, initial_phase=None, seed=42
):
    """ integrate kuramoto """
    np.random.seed(seed)

    if initial_phase is not None:
        repeats = 1

    edge_results = []
    for r in range(repeats):

        if initial_phase is None:
            initial_phase = np.random.random(simplicial_complex.n_edges)

        edge_results.append(
            integrate_edge_kuramoto(
                simplicial_complex,
                initial_phase,
                t_max,
                n_t,
                alpha_1=parameters[0],
                alpha_2=parameters[1],
            )
        )

    return edge_results


def compute_scan(
    simplicial_complex,
    parameter_combinations,
    n_workers=1,
    repeats=20,
    t_max=100,
    n_t=1000,
    initial_phase=None,
):
    """Compute scan"""

    with Pool(n_workers) as pool:
        return list(
            tqdm(
                pool.imap(
                    partial(
                        integrate_kuramoto,
                        simplicial_complex=simplicial_complex,
                        repeats=repeats,
                        t_max=t_max,
                        n_t=n_t,
                        initial_phase=initial_phase,
                    ),
                    parameter_combinations,
                ),
                total=len(parameter_combinations),
            )
        )


def plot_phases(path, filename):
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
    axs = np.flip(axs, axis=0)
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        plt.sca(axs[idx_a1, idx_a2])
        result = results[i][0]
        plt.imshow(
            np.round(result.y + np.pi, 2) % (2 * np.pi) - np.pi,
            origin="lower",
            aspect="auto",
            cmap="twilight_shifted",
            interpolation="nearest",
            extent=(result.t[0], result.t[-1], 0, len(result.y)),
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for idx_a1 in range(len(alpha1)):
        axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
    for idx_a2 in range(len(alpha2)):
        axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

    fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
    fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
    fig.tight_layout()
    plt.savefig(filename, bbox_inches="tight")


def shanahan_metrics(results, alpha1, alpha2, edge_community_assignment):

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


def plot_measures(path, filename, marker=None):

    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
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