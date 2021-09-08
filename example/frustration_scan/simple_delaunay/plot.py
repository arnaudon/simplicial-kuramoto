import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from simplicial_kuramoto.frustration_scan import (
    plot_order_1d,
    compute_order_parameter,
    get_subspaces,
)
from simplicial_kuramoto.plotting import draw_simplicial_complex
from simplicial_kuramoto.frustration_scan import scan_frustration_parameters, proj_subspace
from simplicial_kuramoto import SimplicialComplex

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"
    n_workers = 80
    cmap = cm.get_cmap("Blues", 100)
    frac = 0.5
    gaps_order = []
    edge_flip = [0, 10, 7]
    gaps_curl = []
    n_faces = 6
    s_grads = []
    s_curls = []
    for face in range(n_faces):
        path = folder + f"result_{face}.pkl"
        filename = figure_folder + f"scan_{face}.pdf"
        plot_order_1d(path, filename, n_workers=n_workers, frac=frac)

        Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))
        l1 = Gsc.L1.toarray()
        Gsc.flip_edge_orientation([7])
        print(np.linalg.norm(l1 - Gsc.L1.toarray()))
        w, v = np.linalg.eig(l1)
        print(np.linalg.norm(v - np.linalg.eig(Gsc.L1.toarray())[1], axis=1))
        grad_subspace, curl_subspace, harm_subspace = get_subspaces(Gsc)

        w, v = np.linalg.eig(Gsc.L1.toarray())
        grad_proj = proj_subspace(v.T, grad_subspace)
        curl_proj = proj_subspace(v.T, curl_subspace)
        w_g = w[grad_proj > 1e-12]
        print('g', sorted(w_g))
        w_c = w[curl_proj > 1e-12]
        print('c', sorted(w_c))
        s_grads.append(np.mean(w_g))
        s_curls.append(np.mean(w_c))

        stds = []
        stds_face = []
        for i, a2 in enumerate(alpha2):
            harm_orders = []
            harm_faces = []
            for res in results[i]:
                harm_order, harm_node, harm_face = compute_order_parameter(res.y, Gsc)
                harm_orders.append(harm_order)
                harm_faces.append(harm_face)
            harm_orders = np.mean(harm_orders, axis=0)
            harm_faces = np.mean(harm_faces, axis=0)
            n_min = int(np.shape(res.y)[1] * frac)
            stds.append(np.std(harm_orders[n_min:]))
            stds_face.append(np.std(harm_faces[n_min:]))
        stds = np.array(stds)
        stds_face = np.array(stds_face)
        a2 = alpha2[stds > 1e-5]
        if len(a2) > 0:
            gaps_order.append(a2[0])
        else:
            gaps_order.append(np.pi / 2.0)
        a2_face = alpha2[stds_face > 1e-5]
        if len(a2_face) > 0:
            gaps_curl.append(a2_face[0])
        else:
            gaps_curl.append(np.pi / 2.0)

    plt.figure()
    plt.plot(s_grads, gaps_order, "+")
    plt.xlabel("s_grad")
    plt.ylabel("gaps_order")
    plt.savefig("grad_order.pdf")

    plt.figure()
    plt.plot(s_grads, gaps_curl, "+")
    plt.xlabel("s_grad")
    plt.ylabel("gaps_curl")
    plt.savefig("grad_curl.pdf")

    plt.figure()
    plt.plot(s_curls, gaps_order, "+")
    plt.xlabel("s_curl")
    plt.ylabel("gaps_order")
    plt.savefig("curl_order.pdf")

    plt.figure()
    plt.plot(s_curls, gaps_curl, "+")
    plt.xlabel("s_curl")
    plt.ylabel("gaps_curl")
    plt.savefig("curl_curl.pdf")

    plt.figure()
    plt.plot(gaps_order, gaps_curl, "+")
    plt.xlabel("gaps_order")
    plt.ylabel("gaps_curl")
    plt.savefig("gaps_order_gaps_curl.pdf")

    from scan import make_delaunay

    graph, points = make_delaunay()

    plt.figure()
    ax = plt.gca()

    Gsc = SimplicialComplex(graph=graph)
    Gsc.flip_edge_orientation(edge_flip)

    draw_simplicial_complex(Gsc, filename="graph_curl.pdf", face_colors=gaps_curl)
    draw_simplicial_complex(Gsc, filename="graph_order.pdf", face_colors=gaps_order)
