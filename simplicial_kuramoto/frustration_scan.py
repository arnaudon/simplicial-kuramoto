"""Tools to scan frustration parameters."""
import multiprocessing

import itertools
import logging
import os
import pickle
from functools import partial
from multiprocessing import Pool
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.spatial import distance
from tqdm import tqdm

from simplicial_kuramoto.integrators import integrate_edge_kuramoto
from simplicial_kuramoto.plotting import mod

L = logging.getLogger(__name__)


def _integrate_several_kuramoto(
    parameters,
    simplicial_complex,
    repeats,
    t_max,
    n_t,
    harmonic=False,
):
    """ integrate kuramoto """
    if harmonic:
        grad_subspace, curl_subspace, harm_subspace = get_subspaces(simplicial_complex)

    edge_results = []
    for r in range(repeats):
        initial_phase = np.random.random(simplicial_complex.n_edges)

        edge_results.append(
            integrate_edge_kuramoto(
                simplicial_complex,
                initial_phase,
                t_max,
                n_t,
                alpha_1=parameters[0] * harm_subspace[:, 0] if harmonic else parameters[0],
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
    harmonic=False,
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
                        harmonic=harmonic,
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


def _integrate_several_kuramoto_with_sigma(
    sigma, simplicial_complex, alpha_1, alpha_2, repeats, t_max, n_t, seed=42
):
    """ integrate kuramoto """
    np.random.seed(seed)

    edge_results = []
    for r in range(repeats):

        initial_phase = np.random.random(simplicial_complex.n_edges)

        if not isinstance(alpha_1, (list, np.ndarray)):
            _alpha_1 = np.random.normal(0.0, alpha_1, simplicial_complex.n_edges)
        else:
            _alpha_1 = alpha_1

        edge_results.append(
            integrate_edge_kuramoto(
                simplicial_complex,
                initial_phase,
                t_max,
                n_t,
                alpha_1=_alpha_1,
                alpha_2=alpha_2,
                sigma=sigma,
            )
        )

    return edge_results


def scan_sigma_parameters(
    simplicial_complex,
    sigmas=np.linspace(1, 5, 10),
    alpha1=0.0,
    alpha2=0.0,
    repeats=1,
    n_workers=4,
    t_max=200,
    n_t=1000,
    save=True,
    folder="./results/",
    filename="results_sigma.pkl",
):
    """Scan sigma parameter, if alpha1 is a scalar, it will be std of random alpha_1 vector."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(
                        _integrate_several_kuramoto_with_sigma,
                        simplicial_complex=simplicial_complex,
                        repeats=repeats,
                        t_max=t_max,
                        n_t=n_t,
                        alpha_1=alpha1,
                        alpha_2=alpha2,
                    ),
                    sigmas,
                ),
                total=len(sigmas),
            )
        )

    if save:
        with open(folder + filename, "wb") as f:
            pickle.dump([simplicial_complex, results, sigmas], f)

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


def compute_simplicial_order_parameter(result, harm_subspace, subset=None):
    """Compute simplicial order parmeters, global and a list of partial ones

    Args:
        subset (list): list of bool to select which edge to average over.
    """
    proj = np.zeros_like(result.T)
    mask = np.ones_like(harm_subspace[:, 0], dtype=bool)
    for direction in harm_subspace.T:
        proj += np.outer(result.T.dot(direction), direction)
        mask = mask * (abs(direction) > 1e-10)
    if subset is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            return abs(np.mean(np.exp(1.0j * (result.T / proj)[:, mask * subset]), axis=1))
    else:
        return abs(np.mean(np.exp(1.0j * result.T[:, mask] / proj[:, mask]), axis=1))


def compute_harmonic_projections(result, harm_subspace):
    """Compute cosine similarities along harmonic directions."""
    return [
        ((result / np.linalg.norm(result, axis=0)[np.newaxis]).T.dot(direction)) ** 2
        for direction in harm_subspace.T
    ]


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
    harm_order = np.mean(compute_simplicial_order_parameter(res, harm_subspace))

    grad = grad_slope if np.std(_grad) > eps or grad_slope > eps else np.nan
    curl = curl_slope if np.std(_curl) > eps or curl_slope > eps else np.nan
    harm = harm_slope if np.std(_harm) > eps or harm_slope > eps else np.nan
    return grad, curl, harm, harm_order


def _get_projections_1d(result, frac, eps, grad_subspace, curl_subspace, harm_subspace, Gsc):
    res = result.y
    n_min = int(np.shape(res)[1] * frac)
    res = res[:, n_min:]
    _grad, _curl, _harm, grad_slope, curl_slope, harm_slope = get_projection_slope(
        Gsc, result, grad_subspace, curl_subspace, harm_subspace, n_min
    )
    global_order = compute_simplicial_order_parameter(res, harm_subspace)
    partial_orders = compute_harmonic_projections(res, harm_subspace)
    harm_order = np.mean(global_order)
    harm_partial_orders = np.mean(partial_orders, axis=1)

    grad = grad_slope if np.std(_grad) > eps or grad_slope > eps else np.nan
    curl = curl_slope if np.std(_curl) > eps or curl_slope > eps else np.nan
    harm = harm_slope if np.std(_harm) > eps or harm_slope > eps else np.nan
    return grad, curl, harm, harm_order, harm_partial_orders


def plot_harmonic_order_1d(path, filename, frac=0.5, eps=1e-5, n_workers=4):
    """Plot grad, curl and harm subspaces projection measures."""

    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    grad = []
    curl = []
    harm = []
    harm_order = []
    harm_partial_orders = [[] for _ in range(len(harm_subspace.T))]
    alphas = []
    for i, a2 in tqdm(enumerate(alpha2), total=len(alpha2)):
        for result in results[i]:
            (_grad, _curl, _harm, _harm_order, _harm_partial_order) = _get_projections_1d(
                result, frac, eps, grad_subspace, curl_subspace, harm_subspace, Gsc
            )
            grad.append(_grad)
            curl.append(_curl)
            harm.append(_harm)
            harm_order.append(_harm_order)
            for _i, _harm_partial in enumerate(_harm_partial_order):
                harm_partial_orders[_i].append(_harm_partial)
            alphas.append(a2)

    def _mean(alphas, data):
        df = pd.DataFrame()
        df["alpha"] = alphas
        df["data"] = data
        return df.groupby("alpha").mean().sort_values(by="alpha")

    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    plt.sca(axs[0])

    plt.plot(alphas, grad, ".", c="C0", ms=1)
    grad_df = _mean(alphas, grad)
    plt.plot(grad_df.index, grad_df.data, "-", c="C0", label="grad")

    plt.plot(alphas, curl, ".", c="C1", ms=1)
    curl_df = _mean(alphas, curl)
    plt.plot(curl_df.index, curl_df.data, "-", c="C1", label="curl")

    plt.plot(alphas, harm, ".", c="C2", ms=1)
    harm_df = _mean(alphas, harm)
    plt.plot(harm_df.index, harm_df.data, "-", c="C2", label="harm")
    plt.ylabel("slope")
    plt.legend()
    # plt.grid(True)

    plt.sca(axs[1])
    plt.plot(alphas, harm_order, ".", c="C3", ms=1)
    harm_order_df = _mean(alphas, harm_order)
    plt.plot(harm_order_df.index, harm_order_df.data, "-", c="C3", label="order")
    c = ["C4", "C5"]

    _sum_partial = []
    for i, partial_order in enumerate(harm_partial_orders):
        plt.plot(alphas, partial_order, ".", c=c[i], ms=1)
        partial_order_df = _mean(alphas, partial_order)
        _sum_partial.append(partial_order_df.data.to_numpy())
        plt.plot(partial_order_df.index, partial_order_df.data, "-", c=c[i], label="partial_order")
    plt.plot(partial_order_df.index, np.sum(_sum_partial, axis=0), label="sum of partial")

    plt.axhline(0, lw=0.5, c="k")
    plt.axhline(1, lw=0.5, c="k")
    # axs[1].set_ylim(0, 1.01)
    axs[1].set_xlim(alphas[0], alphas[-1])
    plt.legend()
    # plt.grid(True)
    plt.ylabel("order")
    plt.xlabel(r"$alpha_2$")
    plt.savefig(filename, bbox_inches="tight")


def plot_harmonic_order(path, filename, frac=0.8, eps=1e-5, n_workers=4):
    """Plot grad, curl and harm subspaces projection measures."""

    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    grad = np.empty([len(alpha1), len(alpha2)])
    curl = np.empty([len(alpha1), len(alpha2)])
    harm = np.empty([len(alpha1), len(alpha2)])
    harm_order = np.empty([len(alpha1), len(alpha2)])
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
        for (idx_a1, idx_a2), (_grad, _curl, _harm, _harm_order) in tqdm(
            zip(pairs, _res), total=len(pairs)
        ):
            grad[idx_a1, idx_a2] = _grad
            curl[idx_a1, idx_a2] = _curl
            harm[idx_a1, idx_a2] = _harm
            harm_order[idx_a1, idx_a2] = _harm_order

    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    step1 = alpha1[1] - alpha1[0]
    step2 = alpha1[1] - alpha1[0]
    extent = (alpha2[0] - step2, alpha2[-1] - step2, alpha1[0] - step1, alpha1[-1] - step1)

    def _get_scan_boundary(vec, axis=0):
        if axis == 0:
            a1, a2 = np.meshgrid(alpha1[:-1], alpha2)
        if axis == 1:
            a1, a2 = np.meshgrid(alpha1, alpha2[:-1])
        vec = vec.copy()
        vec[~np.isnan(vec)] = 1
        vec[np.isnan(vec)] = 0
        vec = np.diff(vec, axis=axis) > 0
        return a2[vec.T] - step1 / 2.0, a1[vec.T]

    plt.figure(figsize=(5, 4))
    plt.imshow(harm_order, origin="lower", extent=extent, vmin=0, vmax=1)
    plt.plot(*_get_scan_boundary(grad), c="k", lw=2)
    plt.plot(*_get_scan_boundary(curl), c="r", lw=2, ls="--")
    plt.axis(extent)
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.ylabel(r"$\alpha_1$")
    plt.xlabel(r"$\alpha_2$")
    plt.colorbar(label="Harmonic order", fraction=0.02)
    plt.suptitle(
        f"dim(grad) = {np.shape(grad_subspace)[1]}, dim(curl) = {np.shape(curl_subspace)[1]}, dim(harm) = {np.shape(harm_subspace)[1]}",
        fontsize=9,
    )
    plt.savefig(filename, bbox_inches="tight")


def plot_projections(path, filename, frac=0.8, eps=1e-5, n_workers=4):
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
        for (idx_a1, idx_a2), (_grad, _curl, _harm, _) in tqdm(zip(pairs, _res), total=len(pairs)):
            grad[idx_a1, idx_a2] = _grad
            curl[idx_a1, idx_a2] = _curl
            harm[idx_a1, idx_a2] = _harm

    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    fig = plt.figure(figsize=(4, 6))

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

    plt.suptitle(
        f"dim(grad) = {np.shape(grad_subspace)[1]}, dim(curl) = {np.shape(curl_subspace)[1]}, dim(harm) = {np.shape(harm_subspace)[1]}",
        fontsize=9,
    )

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
