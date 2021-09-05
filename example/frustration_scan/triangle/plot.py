from simplicial_kuramoto.frustration_scan import plot_projections, plot_harmonic_order

if __name__ == "__main__":
    path = "results/triangle.pkl"

    plot_projections(path, "projections.pdf")
    plot_harmonic_order(path, "order.pdf")
