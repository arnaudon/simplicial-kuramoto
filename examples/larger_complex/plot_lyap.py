"""Plot lyapunov exponent."""
from simplicial_kuramoto.frustration_scan import plot_lyapunov

if __name__ == "__main__":
    plot_lyapunov("results/complex.pkl", filename="lyap.pdf")
