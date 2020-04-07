"""Plotting functions."""
import matplotlib.pyplot as plt


def plot_node_kuramoto(node_results):
    """Basic plot for node kuramoto."""
    plt.figure()
    plt.imshow(node_results.y, aspect="auto")
    plt.xlabel("time")
    plt.ylabel("mode id")
