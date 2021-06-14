import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters

def my_house(size):
    
    if(size<3):
        print("I can't build a house with this ...")
        return
    
    G=nx.Graph()
    
    for i in range(size):
        G.add_edge(i,i+1,weight=1,edge_com=0)
        
    G.add_edge(i+1,0,weight=1,edge_com=0)
    G.add_edge(2,0,weight=1,edge_com=0)
  
    return G

if __name__ == "__main__":

    for house_size in [4,8,16,32,64]:
        G=my_house(house_size)

        Gsc=SimplicialComplex(graph=G, no_faces=False)

        alpha1 = np.linspace(0, np.pi, 40)
        alpha2 = np.linspace(0, np.pi, 40)
        n_repeats = 1

        scan_frustration_parameters(
            Gsc,
            filename="house_"+str(house_size)+".pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=12,
        )

        Gsc.flip_edge_orientation(1)
        Gsc.flip_edge_orientation(2)

        scan_frustration_parameters(
            Gsc_noface,
            filename="house_"+str(house_size)+".pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=12,
        )