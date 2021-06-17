"""Graph generation."""
import networkx as nx
import itertools
import numpy as np
from scipy import spatial


def modular_graph(Nc, Nn, Nie, rando=True, inter_weight=0.5, intra_weight=0.5):
    """
    Produces a modular network with Nc clique modules of size Nn and all connected by Nie edges, in a linear way
        Nc: number of modules
        Nn: number of nodes per module
        Nie: number of edges between modules (added linearly), has to be smaller than Nn(Nn-1)/2
    """

    G = nx.Graph()
    G.add_nodes_from(np.linspace(1, Nc * Nn, Nc * Nn).astype(int).tolist())
    node_assign, edge_assign = {}, {}
    # fully connected modules
    for i in range(Nc):
        for j in range(1, Nn + 1):
            for k in range(j + 1, Nn + 1):
                G.add_edge(
                    (i * Nn) + j,
                    (i * Nn) + k,
                    weight=intra_weight,
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
                        G.add_edge(
                            Neig.tolist()[j] + c1 * Nn,
                            np.roll(Neig, -(nr)).tolist()[j] + c2 * Nn,
                            weight=inter_weight,
                            community=str((c1, c2)),
                        )
    nx.set_node_attributes(G, node_assign, "community")
    return G


def delaunay_with_holes(n_points, centres, radii, n_nodes_hole=20, points=None):
    """Create a delanay mesh with holes."""
    if points is None:
        points = np.random.uniform(0, 1, [n_points, 2])

    x = np.linspace(0, 2 * np.pi, n_nodes_hole + 1)[:-1]
    idx_inside = []
    for i in range(len(centres)):
        points = [p for p in points if np.linalg.norm(p - centres[i]) > radii[i]]
    for i in range(len(centres)):
        points += list(
            np.vstack(
                [
                    centres[i][0] + radii[i] * np.sin(x),
                    centres[i][1] + radii[i] * np.cos(x),
                ]
            ).T
        )
        idx_inside.append(np.arange(len(points) - n_nodes_hole, len(points)))
    points = np.array(points)
    tri = spatial.Delaunay(points)

    edge_list = []
    for t in tri.simplices:
        for edge in itertools.combinations(t, 2):

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

    graph = nx.Graph()
    graph.add_edges_from(edge_list)

    return graph, points
