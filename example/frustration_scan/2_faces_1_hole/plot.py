from simplicial_kuramoto.frustration_scan import plot_projections, plot_harmonic_order
import matplotlib.pyplot as plt
import networkx as nx

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    sizes = [3]#, 4, 5, 6, 7, 8, 9, 10]
    for size in sizes:

        path = folder + "2_triangles_face.pkl"
        filename = figure_folder + "2_triangles_face_proj.pdf"
        plot_projections(path, filename, n_workers=80)
        filename = figure_folder + "2_triangles_face_order.pdf"
        plot_harmonic_order(path, filename, n_workers=80, frac=0.95)
        plt.close()
