"""Graph generation."""
import itertools

import networkx as nx
import numpy as np
from scipy import spatial

# pylint: disable=too-many-nested-blocks,too-many-branches


def modular_graph(Nc, Nn, Nie, rando=True, inter_weight=0.5, intra_weight=0.5):
    """Creates a modular network.

    The network is constructed with Nc clique modules of size Nn and all connected
    by Nie edges, in a linear way.

    Args:
        Nc: number of modules
        Nn: number of nodes per module
        Nie: number of edges between modules (added linearly), has to be smaller than Nn(Nn-1)/2
    """

    G = nx.Graph()
    G.add_nodes_from(np.linspace(1, Nc * Nn, Nc * Nn).astype(int).tolist())
    node_assign = {}
    # fully connected modules
    for i in range(Nc):
        for j in range(1, Nn + 1):
            for k in range(j + 1, Nn + 1):
                G.add_edge(
                    (i * Nn) + j,
                    (i * Nn) + k,
                    weight=intra_weight + np.random.normal(0, 0.1),
                    community=str((i, i)),
                )
                node_assign[(i * Nn) + j] = str(i)
                node_assign[(i * Nn) + k] = str(i)

    if rando:
        for i in range(Nc):
            for j in range(i + 1, Nc):
                source = np.random.randint(i * Nn + 1, (i + 1) * Nn, Nie).tolist()
                sink = np.random.randint(j * (Nn + 1) + 1, (j + 1) * Nn, Nie).tolist()
                for e in range(Nie):
                    G.add_edge(source[e], sink[e], weight=inter_weight, community=str((i, j)))

    else:
        Neig = np.linspace(1, Nn, Nn).astype(int)
        if Nie > 0:
            nr, bonus = np.divmod(Nie, Nn)
            for c1 in range(Nc):
                for c2 in range(c1 + 1, Nc):
                    for i in range(nr):
                        for j in range(Nn):
                            G.add_edge(
                                Neig.tolist()[j] + c1 * Nn,
                                np.roll(Neig, -i).tolist()[j] + c2 * Nn,
                                weight=inter_weight,
                                community=str((c1, c2)),
                            )
                    for j in range(bonus):
                        a = Neig.tolist()[j] + c1 * Nn
                        b = np.roll(Neig, -(nr)).tolist()[j] + c2 * Nn
                        # this trick below is to get a symetric 3-module graph
                        if c1 == 0 and b == 11:
                            b = 12
                        elif c1 == 0 and b == 12:  # and ok:
                            b = 11
                        G.add_edge(
                            a,
                            b,
                            weight=inter_weight + np.random.normal(0, 0.1),
                            community=str((c1, c2)),
                        )
    nx.set_node_attributes(G, node_assign, "community")
    return G


def ring_of_rings(num_rings, ring_size):
    """Create ring of rings network."""

    G = nx.Graph()
    for i in range(num_rings):
        gc = nx.generators.classic.circulant_graph(ring_size, [1])
        edges = gc.edges()
        for edge in edges:
            edge_ = [(i * ring_size) + x for x in edge]
            G.add_edge(edge_[0], edge_[1], community=i)

        G.add_edge(
            i * ring_size + 1,
            (i + 1) * ring_size % (num_rings * ring_size),
            community=999,
        )
    return G


def delaunay_with_holes(n_points=None, centres=None, radii=None, n_nodes_hole=20, points=None):
    """Create a delanay mesh with holes, if centres=None, no holes will be created."""
    if points is None:
        points = np.random.uniform(0, 1, [n_points, 2])

    x = np.linspace(0, 2 * np.pi, n_nodes_hole + 1)[:-1]
    idx_inside = []
    if centres is not None:
        for i in range(len(centres)):
            points = [p for p in points if np.linalg.norm(p - centres[i]) > radii[i]]
        for i, (centre, radius) in enumerate(zip(centres, radii)):
            points += list(
                np.vstack(
                    [
                        centre[0] + radius * np.sin(x),
                        centre[1] + radius * np.cos(x),
                    ]
                ).T
            )
            idx_inside.append(np.arange(len(points) - n_nodes_hole, len(points)))
    points = np.array(points)
    tri = spatial.Delaunay(points)

    edge_list = []
    for t in tri.simplices:
        for edge in itertools.combinations(t, 2):
            if centres is not None:
                # add edges not touching the hole
                edge0 = any(edge[0] in idx_inside[i] for i in range(len(centres)))
                edge1 = any(edge[1] in idx_inside[i] for i in range(len(centres)))
                if not edge0 or not edge1:
                    edge_list.append(edge)

                # add the edges in the boundary of holes
                for i in range(len(centres)):
                    if edge[0] in idx_inside[i] and edge[1] in idx_inside[i]:
                        if (
                            np.linalg.norm(points[edge[0]] - points[edge[1]])
                            < 2.0 * np.pi * radii[i] / n_nodes_hole
                        ):
                            edge_list.append(edge)
            else:
                edge_list.append(edge)
    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    for n, p in zip(graph.nodes, points):
        graph.nodes[n]["pos"] = p
    return graph, points
