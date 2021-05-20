import logging
from functools import partial
import itertools
import os

import numpy as np
from tqdm import tqdm
import pickle

from multiprocessing import Pool

from simplicial_kuramoto.integrators import integrate_edge_kuramoto


L = logging.getLogger(__name__)


def scan_chimera_parameters(
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
    parameters,
    simplicial_complex,
    repeats,
    t_max,
    n_t,
    initial_phase=None,
):
    """ integrate kuramoto """
    np.random.seed(42) #int(10 * parameters[0]) + int(10 * parameters[1]))

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
    """Compute scan

    Args:

    Returns:

    """

    L.info("Computing %s parameter combinations.", len(parameter_combinations))
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
