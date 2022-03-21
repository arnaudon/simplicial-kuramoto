from simplicial_kuramoto.frustration_scan import plot_projections, plot_order
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for i in range(50):
        path = f"results/square_{i}.pkl"
        plot_projections(path, f"projections_{i:02d}.png")
        plot_order(path, f"order_{i:02d}.png")
        plt.close('all')
