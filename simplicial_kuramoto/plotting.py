"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

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


def draw_simplicial_complex(
    Gsc,
    filename=None,
    with_labels=True,
    face_colors=None,
    face_vmin=0.0,
    face_vmax=np.pi / 2.0,
    face_cmap="Blues",
    edge_color=None,
    edge_cmap="bwr",
    figsize=None,
):
    """Draw a simplicial complex."""
    plt.figure(figsize=figsize or (4, 3))
    ax = plt.gca()
    points = np.array([Gsc.graph.nodes[n]["pos"] for n in Gsc.graph])
    graph = nx.DiGraph(Gsc.edgelist)

    if face_colors is not None:
        cmap = cm.get_cmap(face_cmap)
        norm = colors.Normalize(
            vmin=face_vmin if face_vmin is not None else min(face_colors),
            vmax=face_vmax if face_vmax is not None else max(face_colors),
        )
    if Gsc.faces is not None:
        for i, face in enumerate(Gsc.faces):
            c = cmap(norm(face_colors[i])) if face_colors is not None else "0.8"
            ax.fill(*points[face].T, c=c)
            if with_labels:
                ax.text(*points[face].mean(0), i)

    if face_colors is not None:
        plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm))

    nx.draw(graph, pos=points, node_size=10 if with_labels else 2)
    if edge_color is not None:
        nx.draw_networkx_edges(
            graph,
            pos=points,
            width=2,
            edge_color=edge_color,
            edge_cmap=plt.get_cmap(edge_cmap),
            edge_vmin=-max(abs(edge_color)),
            edge_vmax=max(abs(edge_color)),
            edgelist=Gsc.edgelist
        )
    if with_labels:
        nx.draw_networkx_labels(graph, pos=points)
        nx.draw_networkx_edge_labels(
            graph, pos=points, edge_labels={e: i for i, e in enumerate(Gsc.edgelist)}
        )

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
