"""Plot lyapunov exponent."""
from simplicial_kuramoto.frustration_scan import plot_lyapunov

if __name__ == "__main__":

    folder = "./results/"
    for i in range(3):
        filename = f"complex_{i}"
        path = path = folder + filename + ".pkl"
        plot_lyapunov(path, filename=f"lyap_{i}.pdf")
