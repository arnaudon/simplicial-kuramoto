import numpy as np
import networkx as nx

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import ring_of_rings

if __name__ == "__main__":

    rings = range(3,10)
    ring_size=3
    for i,num_rings in enumerate(rings):
        
        G = ring_of_rings(num_rings, ring_size)
        Gsc = SimplicialComplex(graph=G, no_faces=False)

        alpha1 = np.linspace(0, 2.2, 100)
        alpha2 = np.linspace(0, np.pi, 100)
        n_repeats = 1
        t_max = 100
        n_t = 100
        n_workers = 80

        scan_frustration_parameters(
            Gsc,
            filename="roft_{}_{}.pkl".format(ring_size,num_rings),
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=n_workers,
            t_max=t_max,
            n_t=n_t,
        )


