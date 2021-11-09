"""Plot frustration scan."""
from simplicial_kuramoto.frustration_scan import plot_order_1d

if __name__ == "__main__":

    plot_order_1d(path="results/complex.pkl", filename="scan_graph_larger.pdf", with_std=True)
