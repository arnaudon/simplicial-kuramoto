from simplicial_kuramoto.frustration_scan import plot_harmonic_order_1d

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"
    n_workers = 80
    n_points = 6
    face = 1
    path = folder + f"simple_hole_{n_points}_{face}.pkl"
    filename = figure_folder + f"simple_hole_{n_points}_{face}.pdf"
    plot_harmonic_order_1d(path, filename, n_workers=n_workers, frac=0.95)
