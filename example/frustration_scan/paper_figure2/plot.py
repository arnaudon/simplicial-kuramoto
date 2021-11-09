import pickle
from simplicial_kuramoto.frustration_scan import plot_order_1d
from simplicial_kuramoto.plotting import draw_simplicial_complex
import matplotlib.pyplot as plt

if __name__ == "__main__":

    folder = "./results/"
    plot_order_1d(path='results/Fig_2_example_large2.pkl', filename="scan_graph_larger2.pdf", with_std=True)
    plt.show()
    plot_order_1d(path='results/Fig_2_example_large.pkl', filename="scan_graph_larger.pdf", with_std=True)
    for i in range(1, 6):
        filename = f"Fig_2_example_1_{i}"
        path = path = folder + filename + ".pkl"

        Gsc, results, alpha1, alpha2 = pickle.load(open(path, "rb"))

        plot_order_1d(path=path, filename=filename + ".pdf")
        draw_simplicial_complex(Gsc, filename=filename + "_graph.pdf")
