"""Representation of simplicial complex"""
import networkx as nx
import numpy as np
import scipy as sc


class SimplicialComplex:
    """Class representing a simplicial complex."""

    def __init__(self, graph=None, faces=None):
        """Initialise the class.

        Args: 
            graph (networkx): original graph to consider
            faces (?): information on which faces we have
        """
        if graph is None:
            raise Exception("Please provide at least a graph")

        self.graph = graph
        self.faces = faces

        self.n_nodes = len(self.graph)
        self.n_edges = len(self.graph.edges)

        self.create_node_incidence_matrix()
        self.create_edge_incidence_matrix()
        self.create_edge_weights_matrix()
        self.degree = np.array([len(self.graph[u]) for u in self.graph])

    def create_edge_weights_matrix(self):
        """Create edge weight matrix."""
        if "weight" in list(self.graph.edges)[0]:
            edge_weights = [self.graph[u][v]["weight"] for u, v in self.graph.edges]
        else:
            edge_weights = np.ones(len(self.graph.edges))
        self.edge_weight_matrix = sc.sparse.spdiags(
            edge_weights, 0, self.n_edges, self.n_edges
        )

    def create_node_incidence_matrix(self):
        """Create node incidence matrix."""
        self.node_incidence_matrix = nx.incidence_matrix(self.graph, oriented=True).T

    def create_edge_incidence_matrix(self):
        """Create edge incidence matrix."""
        if self.faces is not None:
            raise Exception("Not working yet!")
            # self.edge_incidence_matrix = nx.incidence_matrix(self.graph, oriented = True).T
        else:
            self.edge_incidence_matrix = None

    def flip_edge_orientation(self, edge_index):
        """Flip the orientation of an edge."""
        self.node_incidence_matrix[edge_index] *= -1
