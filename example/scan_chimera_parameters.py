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


#%%

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
Gsc_noface = SimplicialComplex(graph=G, no_faces=True)

alpha1 = np.linspace(0,np.pi,30)
alpha2 = np.linspace(0,np.pi,30)
n_repeats = 20

results = scan_chimera_parameters(Gsc,filename='two_comms_three_nodes.pkl',alpha1=alpha1,alpha2=alpha2,repeats=n_repeats,)
results = scan_chimera_parameters(Gsc_noface,filename='two_comms_three_nodes_nofaces.pkl',alpha1=alpha1,alpha2=alpha2,repeats=n_repeats,)



#%%



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
Gsc_noface = SimplicialComplex(graph=G, no_faces=True)

alpha1 = np.linspace(0,np.pi,30)
alpha2 = np.linspace(0,np.pi,30)
n_repeats = 20

results = scan_chimera_parameters(Gsc,filename='two_comms_four_nodes.pkl',alpha1=alpha1,alpha2=alpha2,repeats=n_repeats,)
results = scan_chimera_parameters(Gsc_noface,filename='two_comms_four_nodes_nofaces.pkl',alpha1=alpha1,alpha2=alpha2,repeats=n_repeats,)


#%%


from simplicial_kuramoto.graph_generator import modular_graph

Nn = 5
Nie = int(Nn*(Nn-1)/2)
inter_edges = np.linspace(1,Nie,Nie)


n_inits = 10
alpha1 = np.linspace(0,np.pi,30)
alpha2 = [0]
n_repeats = 20

results_dict = {}
for n_inter_edges in inter_edges:
    
    results_inits = []
    for init in range(n_inits):
        
        g = modular_graph(2, Nn, int(n_inter_edges), rando=True, inter_weight=1, intra_weight=1)
        while not nx.is_connected(g):
            g = modular_graph(2, Nn, int(n_inter_edges), rando=True, inter_weight=1, intra_weight=1)
   
            
        results = scan_chimera_parameters(Gsc,save=False,alpha1=alpha1,alpha2=alpha2,repeats=n_repeats)
        results_inits.append(results)
            
    results_dict[p_out] = results_inits










