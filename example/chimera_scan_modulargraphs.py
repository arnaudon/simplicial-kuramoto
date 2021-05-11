import numpy as np
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
from simplicial_kuramoto.plotting import *

from chimera_scan import scan_chimera_parameters



folder = './results/'


Nn = 5
Nie = int(Nn*(Nn-1)/2)
inter_edges = np.linspace(1,Nie,Nie,dtype=np.int64)


n_inits = 10
alpha1 = np.linspace(0,np.pi,30)
alpha2 = [0]
n_repeats = 20

results_dict = {}
results_dict_nofaces = {}
for n_inter_edges in inter_edges:
    
    results_inits = []
    results_inits_nofaces = []
    for init in range(n_inits):
        
        g = modular_graph(2, Nn, n_inter_edges, rando=True, inter_weight=1, intra_weight=1)
        while not nx.is_connected(g):
            g = modular_graph(2, Nn, n_inter_edges, rando=True, inter_weight=1, intra_weight=1)
   
        Gsc = SimplicialComplex(graph=g, no_faces=False)
        results = scan_chimera_parameters(Gsc,save=False,alpha1=alpha1,alpha2=alpha2,repeats=n_repeats)
        results_inits.append((Gsc,results))

        Gsc = SimplicialComplex(graph=g, no_faces=True)
        results = scan_chimera_parameters(Gsc,save=False,alpha1=alpha1,alpha2=alpha2,repeats=n_repeats)        
        results_inits_nofaces.append((Gsc,results))

            
    results_dict[n_inter_edges] = results_inits
    results_dict_nofaces[n_inter_edges] = results_inits_nofaces



with open(folder+'modular_chimera_scan.pkl', 'wb') as f:
    pickle.dump([results_dict, results_dict_nofaces], f)







