import numpy as np
import networkx as nx
import random

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

def get_delauney_holes_multi(n_points, centre_holes,radius,points=None):


    if points is None:
        x = np.random.rand(n_points)
        y = np.random.rand(n_points)
        points = np.vstack([x,y]).T

    tri = Delaunay(points)

    edge_list = []
    
    
    idx_inside=np.empty([0], dtype=int)
    for i in range(centre_holes.shape[0]):
        idx_inside=np.hstack([idx_inside,encloses([centre_holes[i]],points,radius)[1]])
    
    for t in tri.simplices:
        
        if t[0] not in idx_inside and t[1] not in idx_inside:
            edge_list.append([t[0], t[1]])
     
        if t[1] not in idx_inside and t[2] not in idx_inside:
            edge_list.append([t[1], t[2]])

        if t[0] not in idx_inside and t[2] not in idx_inside:
            edge_list.append([t[0], t[2]])   


            
    graph = nx.Graph()
    #graph.add_nodes_from(np.arange(len(points)))    
    graph.add_edges_from(edge_list)
    
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    g = graph.subgraph(Gcc[0])
    
    
    return g, points


def encloses(centre, points, radius):
    inside_hole = (cdist(centre, points, 'euclidean') <= radius)
    idx_inside = np.where(inside_hole )
    
    return idx_inside

if __name__ == "__main__":

    # no hole
    for radius_size in [0.0,0.1,0.2,0.3,0.4]:
        np.random.seed(4444)
        centre_hole_1 = np.array([[0.5,0.5]])
        graph, points = get_delauney_holes_multi(100,centre_hole_1,radius_size)

        Gsc=SimplicialComplex(graph=graph, no_faces=False) 

        alpha1 = np.linspace(0, np.pi, 40)
        alpha2 = np.linspace(0, np.pi, 40)
        n_repeats = 1

        scan_frustration_parameters(
            Gsc,
            filename="Delaunay_one_hole_r_0_"+str(radius_size).split('.')[1]+".pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=12,
        )
 