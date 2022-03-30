import numpy as np

from simplicial_kuramoto import SimplicialComplex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters
from simplicial_kuramoto.graph_generator import delaunay_with_holes
from simplicial_kuramoto.plotting import draw_simplicial_complex

if __name__ == "__main__":
    points = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
    ws = np.linspace(0.1, 1.0, 4)

    for i, w in enumerate(ws):
        graph, points = delaunay_with_holes(points=points)
        for _i, (u, v) in enumerate(graph.edges):
            if _i == 0:
                graph[u][v]["weight"] = 1.0
            else:
                graph[u][v]["weight"] = 1.0

        Gsc = SimplicialComplex(graph=graph)
        if w == 0:
            Gsc = SimplicialComplex(graph=graph, no_faces=True)
            print(Gsc.faces)
        else:
            Gsc = SimplicialComplex(graph=graph, faces=[Gsc.faces[0]], face_weights=[w])
        draw_simplicial_complex(Gsc, "complex.pdf")

        alpha1 = np.linspace(0, 2.5, 80)
        alpha2 = np.linspace(0, np.pi / 2.0, 120)
        n_repeats = 1
        t_max = 1000
        n_t = 500
        n_workers = 80

        scan_frustration_parameters(
            Gsc,
            filename=f"square_half_{i}.pkl",
            alpha1=alpha1,
            alpha2=alpha2,
            repeats=n_repeats,
            n_workers=n_workers,
            t_max=t_max,
            n_t=n_t,
        )
