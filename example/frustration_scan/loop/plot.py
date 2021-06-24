from simplicial_kuramoto.frustration_scan import plot_projections, plot_rqa
import matplotlib.pyplot as plt
import networkx as nx
from scan_loop import make_loop

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    for size in sizes:
        Gsc = make_loop(size)

        path = folder + f"loop_{size}.pkl"
        filename = figure_folder + f"loop_{size}.pdf"
        plot_projections(path, filename)
        plt.close()
