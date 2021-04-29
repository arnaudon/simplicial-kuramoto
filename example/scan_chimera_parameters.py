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


def scan_parameters(Gsc,
                    filename='parameter_scan.pkl',
                    tmax=100,
                    n_t=100,
                    n_alpha1=20,
                    n_alpha2=20,
                    random_seed=None
                    ):
    
    np.random.seed(random_seed)
    initial_phase = np.random.random(Gsc.n_edges)
    
    t_max = 100
    n_t = 100
    
    alpha_1 = np.linspace(0,np.pi,n_alpha1)
    alpha_2 = np.linspace(0,np.pi,n_alpha2)
    
    gms = np.zeros([alpha_1.shape[0],alpha_2.shape[0]])
    chi = np.zeros([alpha_1.shape[0],alpha_2.shape[0]])
    ce = np.zeros([alpha_1.shape[0],alpha_2.shape[0]])
    ceg = np.zeros([alpha_1.shape[0],alpha_2.shape[0]])
    
    for i,a1 in enumerate(tqdm(alpha_1)):
        for j,a2 in enumerate(alpha_2):
            edge_result = integrate_edge_kuramoto(
                Gsc, initial_phase, t_max, n_t, alpha_1=a1, alpha_2=a2
            )
            op=module_order_parameter(edge_result.y,edge_community_assignment)    
            gop, phase_gradient = module_gradient_parameter(edge_result.y, edge_community_assignment)
            si = Shanahan_indices(op)
            gms[i,j] = si[0]
            chi[i,j] = si[1]
            ce[i,j] = coalition_entropy(op)
            ceg[i,j] = coalition_entropy(gop)
            
            
    with open(filename, 'wb') as f:
        pickle.dump([Gsc, gms, chi, ce, ceg], f)
        
    return 

#### 3 nodes in each community

G=nx.Graph()
G.add_edge(0,1,weight=1,edge_com=0)
G.add_edge(1,2,weight=1,edge_com=0)
G.add_edge(2,0,weight=1,edge_com=0)
G.add_edge(3,4,weight=1,edge_com=1)
G.add_edge(4,5,weight=1,edge_com=1)
G.add_edge(5,3,weight=1,edge_com=1)
G.add_edge(2,3,weight=1,edge_com=2)

node_com_dict=dict(zip(list(np.linspace(0,5,6).astype(int)),[0,0,0,1,1,1]))
nx.set_node_attributes(G, node_com_dict, "node_com")
edge_community_assignment=np.array(list(nx.get_edge_attributes(G,'edge_com').values()))
Gsc = SimplicialComplex(graph=G, no_faces=False)
scan_parameters(Gsc,'results_graph_3_3.pkl')






#### 4 nodes in each community

G=nx.Graph()
G.add_edge(0,1,weight=1,edge_com=0)
G.add_edge(1,2,weight=1,edge_com=0)
G.add_edge(2,3,weight=1,edge_com=0)
G.add_edge(3,0,weight=1,edge_com=0)
G.add_edge(1,3,weight=1,edge_com=0)
G.add_edge(0,2,weight=1,edge_com=0)
G.add_edge(4,5,weight=1,edge_com=1)
G.add_edge(5,6,weight=1,edge_com=1)
G.add_edge(6,7,weight=1,edge_com=1)
G.add_edge(7,4,weight=1,edge_com=1)
G.add_edge(5,7,weight=1,edge_com=1)
G.add_edge(4,6,weight=1,edge_com=1)
G.add_edge(3,4,weight=1,edge_com=2)

node_com_dict=dict(zip(list(np.linspace(0,5,6).astype(int)),[0,0,0,0,1,1,1,1]))
nx.set_node_attributes(G, node_com_dict, "node_com")
edge_community_assignment=np.array(list(nx.get_edge_attributes(G,'edge_com').values()))
Gsc = SimplicialComplex(graph=G, no_faces=False)

scan_parameters(Gsc,'results_graph_4_4.pkl')


