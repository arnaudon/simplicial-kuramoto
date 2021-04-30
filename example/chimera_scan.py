
import logging
import time
from collections import defaultdict
from functools import partial
from importlib import import_module
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing

import multiprocessing
import multiprocessing.pool

from simplicial_kuramoto.integrators import integrate_edge_kuramoto


L = logging.getLogger(__name__)



def scan_chimera_parameters(
                            simplicial_complex,
                            alpha1 = np.linspace(0,np.pi,20),
                            alpha2 = np.linspace(0,np.pi,20),
                            repeats=100,
                            n_workers=4,
                            t_max=100,
                            n_t=100,
                            save=True,
                            initial_phase = None,
                            random_seed=None,
                            ):
    
    
    
    parameter_combinations = list(itertools.product(alpha1,alpha2))
    
    results = compute_scan(
                            simplicial_complex,
                            parameter_combinations,
                            n_workers=n_workers,
                            repeats=repeats,
                            t_max=t_max,
                            n_t=n_t,
                            initial_phase=initial_phase,
                            random_seed=random_seed,
                            )
    
    
    return results


def integrate_kuramoto(
                       parameters, 
                       simplicial_complex,
                       repeats,
                       t_max,
                       n_t,
                       initial_phase=None,
                       random_seed=None,
                       ):
    """ integrate kuramoto """
    
    np.random.seed(random_seed)
    
    if initial_phase is None:
        initial_phase = np.random.random(simplicial_complex.n_edges)
    
    edge_results = []
    for r in range(repeats):
        edge_results.append(
            integrate_edge_kuramoto(
                    simplicial_complex, initial_phase, t_max, n_t, alpha_1=parameters[0], alpha_2=parameters[1],
                )
            )
        
    
    return edge_results
    


def compute_scan(
    simplicial_complex,
    parameter_combinations,
    n_workers=1,
    repeats=100,
    t_max=100,
    n_t=100,
    initial_phase=None,
    random_seed=None,
):
    """Compute scan

    Args:

    Returns:
        
    """

    L.info("Computing %s parameter combinations.", len(parameter_combinations))
    
    
    with NestedPool(n_workers) as pool:
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
                        random_seed=random_seed,
                    ),
                    parameter_combinations,
                ),
                total=len(parameter_combinations),
            )
        )




class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):  # pylint: disable=no-self-use
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess
