"""Tools to scan frustration parameters."""
import pandas as pd
import logging
import networkx as nx
from functools import partial
import itertools
import os

import scipy as sc
from scipy.spatial import distance
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

from simplicial_kuramoto.plotting import mod

L = logging.getLogger(__name__)

# Harmonic flow
def plot_harm(path, filename):
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    
    if(Gsc.L1.shape[0]==0 or Gsc.L1.shape[1]==0):
        print('No L1')
    else:
        Harm_Sub_Space=sc.linalg.null_space(Gsc.L1.todense())
    
        if(Harm_Sub_Space.shape[1]==0):
            print('No holes')
        else:
            print('Number of holes:',Harm_Sub_Space.shape[1])
            print('Max number of dimensions:',Harm_Sub_Space.shape[0])

            fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
            axs = np.flip(axs, axis=0)
            for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
                plt.sca(axs[idx_a1, idx_a2])
                result = mod(results[i][0].y)
                plt.plot(result.T.dot(Harm_Sub_Space))
                plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            for idx_a1 in range(len(alpha1)):
                axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
            for idx_a2 in range(len(alpha2)):
                axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

            fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
            fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
            fig.tight_layout()
            plt.savefig(filename, bbox_inches="tight")
    
# Gradient flow
def plot_grad(path, filename):
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    
    if(Gsc.B0.shape[0]==0 or Gsc.B0.shape[1]==0):
        print('Either no node or no edge (or both), check the complex')
    else:
        Grad_Sub_Space=sc.linalg.orth(Gsc.B0.todense())

        if(Grad_Sub_Space.shape[1]==0):
            print('No gradient')
        else:
            print('Dimension of the gradient space:',Grad_Sub_Space.shape[1])
            print('Max number of dimensions:',Grad_Sub_Space.shape[0])

            fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
            axs = np.flip(axs, axis=0)
            for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
                plt.sca(axs[idx_a1, idx_a2])
                result = mod(results[i][0].y)
                plt.plot(result.T.dot(Grad_Sub_Space))
                plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            for idx_a1 in range(len(alpha1)):
                axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
            for idx_a2 in range(len(alpha2)):
                axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

            fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
            fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
            fig.tight_layout()
            plt.savefig(filename, bbox_inches="tight")
    
# Curl flow
def plot_curl(path, filename):
    Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
    
    if(Gsc.B1.shape[0]==0 or Gsc.B1.shape[1]==0):
        print('Either no edge or no face (or both), check the complex')
    else:
        Curl_Sub_Space=sc.linalg.orth(Gsc.B1.T.todense())

        if(Curl_Sub_Space.shape[1]==0):
            print('No curl')
        else:
            print('Dimension of the curl space:',Curl_Sub_Space.shape[1])
            print('Max number of dimensions:',Curl_Sub_Space.shape[0])

            fig, axs = plt.subplots(len(alpha1), len(alpha2), figsize=(len(alpha2), len(alpha1)))
            axs = np.flip(axs, axis=0)
            for i, (idx_a1, idx_a2) in enumerate(itertools.product(range(len(alpha1)), range(len(alpha2)))):
                plt.sca(axs[idx_a1, idx_a2])
                result = mod(results[i][0].y)
                plt.plot(result.T.dot(Curl_Sub_Space))
                plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            for idx_a1 in range(len(alpha1)):
                axs[idx_a1, 0].set_ylabel(f"{np.round(alpha1[idx_a1], 2)}", fontsize=15)
            for idx_a2 in range(len(alpha2)):
                axs[0, idx_a2].set_xlabel(f"{np.round(alpha2[idx_a2], 2)}", fontsize=15)

            fig.text(-0.01, 0.5, "Alpha 1", va="center", rotation="vertical", fontsize=20)
            fig.text(0.5, -0.01, "Alpha 2", ha="center", fontsize=20)
            fig.tight_layout()
            plt.savefig(filename, bbox_inches="tight")