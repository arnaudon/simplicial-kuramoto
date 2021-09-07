"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx


def mod(x):
    """Apply mod to be in [-pi, pi]"""
    return np.round(x + np.pi, 2) % (2 * np.pi) - np.pi


def plot_node_kuramoto(node_results):
    """Basic plot for node kuramoto."""
    plt.figure(figsize=(10, 5))
    plt.imshow(
        mod(node_results.y),
        aspect="auto",
        cmap="twilight_shifted",
        extent=(node_results.t[0], node_results.t[-1], 0, len(node_results.y)),
        interpolation="nearest",
    )
    plt.xlabel("time")
    plt.ylabel("mode id")
    plt.colorbar()


def plot_edge_kuramoto(edge_results):
    """Basic plot for edge kuramoto."""
    plt.imshow(
        mod(edge_results.y),
        origin="lower",
        aspect="auto",
        cmap="twilight_shifted",
        interpolation="nearest",
        extent=(edge_results.t[0], edge_results.t[-1], 0, len(edge_results.y)),
    )
    plt.title("Phases")
    plt.colorbar()


def draw_simplicial_complex(Gsc, filename=None, with_labels=True):
    """Draw a simplicial complex."""
    plt.figure()
    ax = plt.gca()
    points = np.array([Gsc.graph.nodes[n]["pos"] for n in Gsc.graph])
    graph = nx.DiGraph(Gsc.edgelist)

    for i, face in enumerate(Gsc.faces):
        ax.fill(*points[face].T, c="0.8")
        if with_labels:
            ax.text(*points[face].mean(0), i)

    nx.draw(graph, pos=points)
    if with_labels:
        nx.draw_networkx_labels(graph, pos=points)
        nx.draw_networkx_edge_labels(
            graph, pos=points, edge_labels={e: i for i, e in enumerate(Gsc.graph.edges)}
        )

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
