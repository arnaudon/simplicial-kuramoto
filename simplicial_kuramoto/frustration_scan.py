"""Tools to scan frustration parameters."""
import multiprocessing

import itertools
import logging
import os
import pickle
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.spatial import distance
from tqdm import tqdm

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.plotting import mod

L = logging.getLogger(__name__)


def _integrate_several_kuramoto(
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


def scan_frustration_parameters(
    simplicial_complex,
    alpha1=np.linspace(0, np.pi, 10),
    alpha2=np.linspace(0, 2.0, 10),
    repeats=1,
    n_workers=4,
    t_max=200,
    n_t=1000,
    save=True,
    folder="./results/",
    filename="results.pkl",
    initial_phase=None,
):
    """Scan frustration parameters alpha_1 and alpha_2 and save phases."""

    if not os.path.exists(folder):
        os.makedirs(folder)

    parameter_combinations = list(itertools.product(alpha1, alpha2))

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(
                        _integrate_several_kuramoto,
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

    if save:
        with open(folder + filename, "wb") as f:
            pickle.dump([simplicial_complex, results, alpha1, alpha2], f)

    return results


def plot_phases(path, filename):
    """From result of frustration scan, plot a grid of phase trajectories."""
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
    axs = np.flip(axs, axis=0)
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        plt.sca(axs[idx_a1, idx_a2])
        result = results[i][0]
        plt.imshow(
            mod(result.y),
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


def get_subspaces(Gsc):
    """"Get grad, curl and harm subspaces."""
    grad_subspace = sc.linalg.orth(Gsc.N0.todense())
    try:
        curl_subspace = sc.linalg.orth(Gsc.N1s.todense())
    except (ValueError, AttributeError):
        curl_subspace = np.zeros([len(Gsc.graph.edges), 0])

    harm_subspace = sc.linalg.null_space(Gsc.L1.todense())
    return grad_subspace, curl_subspace, harm_subspace


def proj_subspace(vec, subspace):
    """Project a list of vecs to a subspace."""
    proj = np.zeros_like(vec)
    for direction in subspace.T:
        proj += np.outer(vec.dot(direction), direction)
    return np.linalg.norm(proj, axis=1)


def get_projection_slope(
    Gsc, res, grad_subspace=None, curl_subspace=None, harm_subspace=None, n_min=0
):
    """Project result on subspaces."""
    if grad_subspace is None:
        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    time = res.t[n_min:]

    grad = proj_subspace(res.y.T, grad_subspace)[n_min:]
    grad_fit = np.polyfit(time, grad, 1)
    grad -= grad_fit[0] * time + grad_fit[1]
    grad -= np.min(grad)

    curl = proj_subspace(res.y.T, curl_subspace)[n_min:]
    curl_fit = np.polyfit(time, curl, 1)
    curl -= curl_fit[0] * time + curl_fit[1]
    curl -= np.min(curl)

    harm = proj_subspace(res.y.T, harm_subspace)[n_min:]
    harm_fit = np.polyfit(time, harm, 1)
    harm -= harm_fit[0] * time + harm_fit[1]
    harm -= np.min(harm)

    return grad, curl, harm, grad_fit[0], curl_fit[0], harm_fit[0]


def _get_projections(result, frac, eps, grad_subspace, curl_subspace, harm_subspace, Gsc):
    res = result[0].y
    n_min = int(np.shape(res)[1] * frac)
    res = res[:, n_min:]
    _grad, _curl, _harm, grad_slope, curl_slope, harm_slope = get_projection_slope(
        Gsc, result[0], grad_subspace, curl_subspace, harm_subspace, n_min
    )

    grad = grad_slope if np.std(_grad) > eps or grad_slope > eps else np.nan
    curl = curl_slope if np.std(_curl) > eps or curl_slope > eps else np.nan
    harm = harm_slope if np.std(_harm) > eps or harm_slope > eps else np.nan
    return grad, curl, harm


def plot_projections(path, filename, frac=0.2, eps=1e-3, n_workers=4):
    """Plot grad, curl and harm subspaces projection measures."""

    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    grad = np.empty([len(alpha1), len(alpha2)])
    curl = np.empty([len(alpha1), len(alpha2)])
    harm = np.empty([len(alpha1), len(alpha2)])
    pairs = list(itertools.product(range(len(alpha1)), range(len(alpha2))))

    _eval = partial(
        _get_projections,
        frac=frac,
        eps=eps,
        grad_subspace=grad_subspace,
        curl_subspace=curl_subspace,
        harm_subspace=harm_subspace,
        Gsc=Gsc,
    )
    with multiprocessing.Pool(n_workers) as pool:
        _res = pool.imap(_eval, results, chunksize=max(1, int(0.1 * len(results) / n_workers)))
        for (idx_a1, idx_a2), (_grad, _curl, _harm) in tqdm(zip(pairs, _res), total=len(pairs)):
            grad[idx_a1, idx_a2] = _grad
            curl[idx_a1, idx_a2] = _curl
            harm[idx_a1, idx_a2] = _harm

    fig = plt.figure(figsize=(4, 6))

    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    plt.suptitle(
        f"dim(grad) = {np.shape(grad_subspace)[1]}, dim(curl) = {np.shape(curl_subspace)[1]}, dim(harm) = {np.shape(harm_subspace)[1]}",
        fontsize=9,
    )

    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    step1 = alpha1[1] - alpha1[0]
    step2 = alpha1[1] - alpha1[0]
    extent = (alpha2[0] - step2, alpha2[-1] - step2, alpha1[0] - step1, alpha1[-1] - step1)

    plt.sca(axs[0])
    plt.imshow(grad, origin="lower", extent=extent, vmin=0)
    plt.axis(extent)
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.ylabel(r"$\alpha_1$")
    plt.colorbar(label="Gradient slope", fraction=0.02)

    plt.sca(axs[1])
    plt.imshow(curl, origin="lower", extent=extent, vmin=0)
    plt.ylabel(r"$\alpha_1$")
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.axis(extent)
    plt.colorbar(label="Curl slope", fraction=0.02)

    plt.sca(axs[2])
    plt.imshow(harm, origin="lower", extent=extent, vmin=0)
    plt.axis(extent)
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.ylabel(r"$\alpha_1$")
    plt.xlabel(r"$\alpha_2$")
    plt.colorbar(label="Harmonic slope", fraction=0.02)

    fig.tight_layout()

    plt.savefig(filename, bbox_inches="tight")


def rec_plot(s, eps=0.1, steps=10):
    """Compute recurence plot.

    Adapted from: https://github.com/laszukdawid/recurrence-plot/blob/master/plot_recurrence.py
    """
    return distance.squareform(np.clip(np.floor_divide(distance.pdist(s.T), eps), 0, steps))


def plot_recurences(path, filename, eps=0.1, steps=10):
    """Plot grid of recurence plots."""
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
    axs = np.flip(axs, axis=0)
    for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
        plt.sca(axs[idx_a1, idx_a2])
        result = mod(results[i][0].y)
        plt.imshow(
            rec_plot(result, eps=eps, steps=steps),
            origin="lower",
            aspect="auto",
            cmap="Blues_r",
            interpolation="nearest",
            extent=(
                results[i][0].t[0],
                results[i][0].t[-1],
                results[i][0].t[0],
                results[i][0].t[-1],
            ),
            vmin=0,
            vmax=steps,
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


def _rqa_comp(X):
    """Compute rqa with pyunicorn."""
    from pyunicorn.timeseries.recurrence_plot import RecurrencePlot

    return RecurrencePlot(X.T, recurrence_rate=0.1, metric="supremum", silence_level=2)


def plot_rqa(path, filename, frac=0.2, min_rr=0.9, n_steps=5):
    """Plot recurence data with pyrqa."""
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

    rr = np.empty([len(alpha1), len(alpha2)])
    det = np.empty([len(alpha1), len(alpha2)])
    diag = np.empty([len(alpha1), len(alpha2)])
    lam = np.empty([len(alpha1), len(alpha2)])
    pairs = list(itertools.product(range(len(alpha1)), range(len(alpha2))))
    for i, (idx_a1, idx_a2) in tqdm(enumerate(pairs), total=len(pairs)):
        result = mod(results[i][0].y)[:, ::n_steps]
        rqa_res = _rqa_comp(result[:, int(np.shape(result)[1] * frac) :])
        rr[idx_a1, idx_a2] = rqa_res.recurrence_rate()
        det[idx_a1, idx_a2] = rqa_res.determinism()
        diag[idx_a1, idx_a2] = rqa_res.average_diaglength()
        lam[idx_a1, idx_a2] = rqa_res.laminarity()

    fig, axs = plt.subplots(2, 2)
    extent = (alpha2[0], alpha2[-1], alpha1[0], alpha1[-1])

    # mask stationary state
    mask = rr < 0.001  # min_rr

    rr[mask] = np.nan
    cm = axs[0, 0].imshow(rr, origin="lower", extent=extent, aspect="auto")
    axs[0, 0].set_title("Recurrence rate")
    plt.colorbar(cm, ax=axs[0, 0])

    det[mask] = np.nan
    cm = axs[0, 1].imshow(det, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(cm, ax=axs[0, 1])
    axs[0, 1].set_title("Determinism")

    diag[mask] = np.nan
    cm = axs[1, 0].imshow(diag, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(cm, ax=axs[1, 0])
    axs[1, 0].set_title("Average diagonal length")

    lam[mask] = np.nan
    cm = axs[1, 1].imshow(lam, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(cm, ax=axs[1, 1])
    axs[1, 1].set_title("Laminarity")

    fig.tight_layout()

    fig.text(0.5, 0.0, "Alpha 2", ha="center")
    fig.text(0.0, 0.5, "Alpha 1", va="center", rotation="vertical")

    plt.savefig(filename, bbox_inches="tight")
