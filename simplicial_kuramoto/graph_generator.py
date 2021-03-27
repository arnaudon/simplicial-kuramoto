"""Graph generation."""
import networkx as nx
import numpy as np


def modular_graph(Nc, Nn, Nie, rando=True, inter_weight=0.5, intra_weight=0.5):
    """
    Produces a modular network with Nc clique modules of size Nn and all connected by Nie edges, in a linear way
        Nc: number of modules
        Nn: number of nodes per module
        Nie: number of edges between modules (added linearly), has to be smaller than Nn(Nn-1)/2
    """

    G = nx.Graph()
    G.add_nodes_from(np.linspace(1, Nc * Nn, Nc * Nn).astype(int).tolist())

    # fully connected modules
    for i in range(Nc):
        for j in range(1, Nn + 1):
            for k in range(j + 1, Nn + 1):
                G.add_edge((i * Nn) + j, (i * Nn) + k, weight=intra_weight)

    if rando:
        for i in range(Nc):
            for j in range(i + 1, Nc):
                source = np.random.randint(i * Nn + 1, (i + 1) * Nn, Nie).tolist()
                sink = np.random.randint(j * (Nn + 1) + 1, (j + 1) * Nn, Nie).tolist()
                for e in range(Nie):
                    G.add_edge(source[e], sink[e], weight=inter_weight)

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
                            )
                    for j in range(bonus):
                        G.add_edge(
                            Neig.tolist()[j] + c1 * Nn,
                            np.roll(Neig, -(nr)).tolist()[j] + c2 * Nn,
                            weight=inter_weight,
                        )

    return G
