from simplicial_kuramoto.frustration_scan import plot_projections, plot_harmonic_order

if __name__ == "__main__":
    folder = "./results/"
    figure_folder = "figures/"

    radii = [0.1, 0.2, 0.3, 0.4]
    for radius in radii:
        path = folder + f"delaunay_one_hole_{radius}.pkl"
        filename = figure_folder + f"delaunay_one_hole_{radius}_proj.pdf"
        plot_projections(path, filename)
        filename = figure_folder + f"delaunay_one_hole_{radius}_order.pdf"
        plot_harmonic_order(path, filename)
        path = folder + f"delaunay_one_hole_{radius}_harmonic.pkl"
        filename = figure_folder + f"delaunay_one_hole_{radius}_harmonic_proj.pdf"
        plot_projections(path, filename)
        filename = figure_folder + f"delaunay_one_hole_{radius}_harmonic_order.pdf"
        plot_harmonic_order(path, filename)
