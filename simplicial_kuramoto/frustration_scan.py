"""Tools to scan frustration parameters."""
import itertools
import logging
import multiprocessing
import os
import pickle
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import nolds
import numpy as np
import pandas as pd
import scipy as sc
from tqdm import tqdm

from simplicial_kuramoto.integrators import integrate_edge_kuramoto

L = logging.getLogger(__name__)


def _integrate_several_kuramoto(
    parameters,
    simplicial_complex,
    repeats,
    t_max,
    n_t,
    harmonic=False,
):
    """Integrate several Kuramotos for parallel computations."""
    if harmonic:
        harm_subspace = get_subspaces(simplicial_complex)[2]

    edge_results = []
    for _ in range(repeats):
        initial_phase = np.random.random(simplicial_complex.n_edges)

        edge_results.append(
            integrate_edge_kuramoto(
                simplicial_complex,
                initial_phase,
                t_max,
                n_t,
                alpha_1=parameters[0] * harm_subspace[:, 0] if harmonic else parameters[0],
                alpha_2=parameters[1],
                disable_tqdm=True,
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
    """Scan frustration parameters alpha_1 and alpha_2.

    Args:
        simplicial_complex (SimplicialComplex): simplicial complex to use
        alpha1 (array): alpha1 values to scan
        alpha2 (array): alpha2 values to scan
        repeats (int): number of repeat of same point with random initial conditions
        n_workers (int): number of workers for multiprocessing
        t_max (float): integration time
        n_t (int): number of timepoints
        save (bool): save results in a picle
        folder (str): folder to save results
        filename (str): name of pickle file
        harmonic (bool): to use a harmonic alpha1 vector scaled by given alpha1

    Returns:
        results of the scan

    """
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


def get_subspaces(Gsc):
    """ "Get grad, curl and harm subspaces from simplicial complex."""
    grad_subspace = sc.linalg.orth(Gsc.N0.todense())
    try:
        curl_subspace = sc.linalg.orth(Gsc.N1s.todense())
    except (ValueError, AttributeError):
        curl_subspace = np.zeros([len(Gsc.graph.edges), 0])

    harm_subspace = sc.linalg.null_space(Gsc.L1.todense())
    return grad_subspace, curl_subspace, harm_subspace


def proj_subspace(vec, subspace):
    """Project a list of vecs to a given subspace (from get_subspaces)."""
    proj = np.zeros_like(vec)
    for direction in subspace.T:
        proj += np.outer(vec.dot(direction), direction)
    return np.linalg.norm(proj, axis=1)


def compute_node_order_parameter(result, Gsc):
    """Compute the node Kuramoto order parameter."""
    w1_inv = 1.0 / np.diag(Gsc.W1.toarray())
    return w1_inv.dot(np.cos(Gsc.N0.dot(result))) / w1_inv.sum()


def compute_order_parameter(result, Gsc, subset=None):
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


def get_projection_fit(
    Gsc, res, grad_subspace=None, curl_subspace=None, harm_subspace=None, n_min=0
):
    """Project result on subspaces and compute linear fit."""
    if grad_subspace is None:
        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)
    time = res.t[n_min:]

    grad = proj_subspace(res.y.T, grad_subspace)[n_min:]
    grad_fit = np.polyfit(time, grad, 1)

    curl = proj_subspace(res.y.T, curl_subspace)[n_min:]
    curl_fit = np.polyfit(time, curl, 1)

    harm = proj_subspace(res.y.T, harm_subspace)[n_min:]
    harm_fit = np.polyfit(time, harm, 1)

    return grad, curl, harm, grad_fit, curl_fit, harm_fit


def _get_projections(result, frac, eps, grad_subspace, curl_subspace, harm_subspace, Gsc):
    res = result[0].y
    n_min = int(np.shape(res)[1] * frac)
    res = res[:, n_min:]
    _grad, _curl, _harm, grad_slope, curl_slope, harm_slope = get_projection_fit(
        Gsc, result[0], grad_subspace, curl_subspace, harm_subspace, n_min
    )
    grad_slope = grad_slope[0]
    curl_slope = curl_slope[0]
    harm_slope = harm_slope[0]

    harm_order = np.mean(compute_order_parameter(res, Gsc)[0])

    grad = grad_slope if np.std(_grad) > eps or grad_slope > eps else np.nan
    curl = curl_slope if np.std(_curl) > eps or curl_slope > eps else np.nan
    harm = harm_slope if np.std(_harm) > eps or harm_slope > eps else np.nan
    return grad, curl, harm, harm_order


def plot_lyapunov(path, filename="lyap.pdf", nolds_kwargs=None):
    """Compute and plot largest lyapunov exponents.

    The computation is based on nolds package, and yse nolds_kwargs to configure it.
    """
    if nolds_kwargs is None:
        nolds_kwargs = {"trajectory_len": 10, "lag": 30, "min_tsep": 20, "fit": "poly"}

    with open(path, "rb") as pick:
        Gsc, results, _, alpha2 = pickle.load(pick)

    lyaps = []
    for i in tqdm(range(Gsc.n_edges)):
        lyapunov = []
        for result in results[1:-1]:
            lyap = []
            for res in result:
                lyap.append(nolds.lyap_r(res.y[i], **nolds_kwargs))
            lyapunov.append(np.mean(lyap))
        lyaps.append(lyapunov)
    lyaps = np.array(lyaps)

    plt.figure(figsize=(4, 2))
    plt.plot(alpha2[1:-1], np.mean(lyaps, axis=0), "k-", lw=2)
    plt.fill_between(
        alpha2[1:-1],
        np.percentile(lyaps, 25, axis=0),
        np.percentile(lyaps, 75, axis=0),
        color="k",
        alpha=0.5,
    )
    plt.gca().set_xlim(0, np.pi / 2)
    plt.gca().set_ylim(-0.002, 0.027)
    plt.axhline(0, c="k", ls="--", lw=0.5)
    plt.xlabel(r"$\alpha_2$")
    plt.ylabel(r"mean($\lambda$)")
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def _get_projections_1d(result, frac, eps, grad_subspace, curl_subspace, harm_subspace, Gsc):
    res = result.y
    n_min = int(np.shape(res)[1] * frac)
    res = res[:, n_min:]
    _grad, _curl, _harm, grad_slope, curl_slope, harm_slope = get_projection_fit(
        Gsc, result, grad_subspace, curl_subspace, harm_subspace, n_min
    )
    grad_slope = grad_slope[0]
    curl_slope = curl_slope[0]
    harm_slope = harm_slope[0]

    harm_order = compute_order_parameter(res, Gsc)[0]
    mean_harm_order = np.mean(harm_order)
    std_harm_order = np.std(harm_order)

    grad = grad_slope if np.std(_grad) > eps or grad_slope > eps else np.nan
    curl = curl_slope if np.std(_curl) > eps or curl_slope > eps else np.nan
    harm = harm_slope if np.std(_harm) > eps or harm_slope > eps else np.nan
    return grad, curl, harm, mean_harm_order, std_harm_order


def plot_order_1d(
    path, filename, frac=0.5, eps=1e-5, with_std=False
):  # pylint: disable=too-many-statements
    """Plot order and projection with fixed alpha1."""

    with open(path, "rb") as pick:
        Gsc, results, _, alpha2 = pickle.load(pick)
    grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

    grad = []
    curl = []
    harm = []
    mean_harm_order = []
    std_harm_order = []
    alphas = []
    for i, a2 in tqdm(enumerate(alpha2), total=len(alpha2)):
        for result in results[i]:
            (_grad, _curl, _harm, _mean_harm_order, _std_harm_order) = _get_projections_1d(
                result, frac, eps, grad_subspace, curl_subspace, harm_subspace, Gsc
            )
            grad.append(_grad)
            curl.append(_curl)
            harm.append(_harm)
            mean_harm_order.append(_mean_harm_order)
            std_harm_order.append(_std_harm_order)
            alphas.append(a2)

    def _mean(alphas, data):
        df = pd.DataFrame()
        df["alpha"] = alphas
        df["data"] = data
        return df.groupby("alpha").mean().sort_values(by="alpha")

    def _std(alphas, data):
        df = pd.DataFrame()
        df["alpha"] = alphas
        df["data"] = data
        return df.groupby("alpha").std().sort_values(by="alpha")

    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    plt.sca(axs[0])

    grad_df = _mean(alphas, grad)
    if with_std:
        std_grad_df = _std(alphas, grad)
        plt.errorbar(grad_df.index, grad_df.data, yerr=std_grad_df.data, c="C0")
    plt.plot(grad_df.index, grad_df.data, "-", c="C0", label="grad")

    curl_df = _mean(alphas, curl)
    if with_std:
        std_curl_df = _std(alphas, curl)
        plt.errorbar(curl_df.index, curl_df.data, yerr=std_curl_df.data, c="C1")
    plt.plot(curl_df.index, curl_df.data, "-", c="C1", label="curl")

    harm_df = _mean(alphas, harm)
    if with_std:
        std_harm_df = _std(alphas, harm)
        plt.errorbar(harm_df.index, harm_df.data, yerr=std_harm_df.data, c="C2")
    plt.plot(harm_df.index, harm_df.data, "-", c="C2", label="harm")
    plt.ylabel("slope")
    plt.legend()

    plt.sca(axs[1])
    harm_order_df = _mean(alphas, mean_harm_order)
    if with_std:
        std_harm_df = _std(alphas, mean_harm_order)
        plt.errorbar(harm_order_df.index, harm_order_df.data, yerr=std_harm_df.data, c="C3")
    plt.plot(harm_order_df.index, harm_order_df.data, "-", c="C3", label="mean(order)")

    plt.ylabel("mean(order)")
    plt.legend(loc="upper right")
    plt.xlabel(r"$\alpha_2$")
    plt.twinx()
    harm_order_df = _mean(alphas, std_harm_order)
    if with_std:
        std_harm_df = _std(alphas, std_harm_order)
        plt.errorbar(harm_order_df.index, harm_order_df.data, yerr=std_harm_df.data, c="C4")
    plt.plot(harm_order_df.index, harm_order_df.data, "-", c="C4", label="std(order)")
    plt.gca().set_ylim(-0.01, max(max(std_harm_order), 0.1))
    axs[1].set_xlim(alphas[0], alphas[-1])
    plt.legend(loc="upper left")
    plt.ylabel("std(order)")
    plt.savefig(filename, bbox_inches="tight")


def plot_order(path, filename, frac=0.5, eps=1e-5, n_workers=4, with_proj=False):
    """Plot order scan."""

    with open(path, "rb") as pick:
        Gsc, results, alpha1, alpha2 = pickle.load(pick)
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
    plt.imshow(harm_order, origin="lower", extent=extent, aspect="auto")
    if with_proj:
        plt.plot(*_get_scan_boundary(grad), c="k", lw=1)
        plt.plot(*_get_scan_boundary(curl), c="r", lw=1, ls="--")
    plt.axis(extent)
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.ylabel(r"$\alpha_1$")
    plt.xlabel(r"$\alpha_2$")
    plt.colorbar(label="Order", fraction=0.02)

    ng = np.shape(grad_subspace)[1]
    nc = np.shape(curl_subspace)[1]
    nh = np.shape(harm_subspace)[1]
    plt.suptitle(f"dim(grad) = {ng}, dim(curl) = {nc}, dim(harm) = {nh}", fontsize=9)

    plt.savefig(filename, bbox_inches="tight")


def plot_projections(path, filename, frac=0.8, eps=1e-5, n_workers=4):
    """Plot grad, curl and harm subspaces projection measures."""
    with open(path, "rb") as pick:
        Gsc, results, alpha1, alpha2 = pickle.load(pick)

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

    fig = plt.figure(figsize=(5, 8))

    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    step1 = alpha1[1] - alpha1[0]
    step2 = alpha1[1] - alpha1[0]
    extent = (alpha2[0] - step2, alpha2[-1] - step2, alpha1[0] - step1, alpha1[-1] - step1)

    plt.sca(axs[0])

    plt.imshow(grad, origin="lower", extent=extent, vmin=0, aspect="auto")
    plt.axis(extent)
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.ylabel(r"$\alpha_1$")
    plt.colorbar(label="Gradient slope", fraction=0.02)

    plt.sca(axs[1])
    plt.imshow(curl, origin="lower", extent=extent, vmin=0, aspect="auto")
    plt.ylabel(r"$\alpha_1$")
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.axis(extent)
    plt.colorbar(label="Curl slope", fraction=0.02)

    plt.sca(axs[2])
    plt.imshow(harm, origin="lower", extent=extent, vmin=0, aspect="auto")
    plt.axis(extent)
    plt.axhline(1, ls="--", c="k", lw=0.5)
    plt.ylabel(r"$\alpha_1$")
    plt.xlabel(r"$\alpha_2$")
    plt.colorbar(label="Harmonic slope", fraction=0.02)

    ng = np.shape(grad_subspace)[1]
    nc = np.shape(curl_subspace)[1]
    nh = np.shape(harm_subspace)[1]
    plt.suptitle(f"dim(grad) = {ng}, dim(curl) = {nc}, dim(harm) = {nh}", fontsize=9)

    fig.tight_layout()

    plt.savefig(filename, bbox_inches="tight")
