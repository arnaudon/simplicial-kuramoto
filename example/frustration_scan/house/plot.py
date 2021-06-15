from simplicial_kuramoto.frustration_scan import plot_projections, plot_rqa
import matplotlib.pyplot as plt
from scan_house_size import my_house
import networkx as nx

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    for size in sizes:
        Gsc = my_house(size)
        plt.figure()
        nx.draw(Gsc.graph)
        plt.savefig(figure_folder + f"house_{size}_graph.pdf")
        plt.close()

        path = folder + f"house_{size}.pkl"
        filename = figure_folder + f"house_{size}_proj.pdf"
        plot_projections(path, filename)
        plt.close()

        path = folder + f"house_{size}_flip.pkl"
        filename = figure_folder + f"house_{size}_flip_proj.pdf"
        plot_projections(path, filename)
        plt.close()
