"""Representation of simplicial complex"""
import networkx as nx
import numpy as np
import scipy as sc


class SimplicialComplex:
    """Class representing a simplicial complex."""

    def __init__(self, graph=None, faces=None, no_faces=False, verbose=True):
        """Initialise the class.

        Args: 
            graph (networkx): original graph to consider
            faces (list): list of faces, each element is a list of ordered 3 nodes
        """
        if graph is None:
            raise Exception("Please provide at least a graph")

        self.graph = graph
        self.n_nodes = len(self.graph)
        self.n_edges = len(self.graph.edges)
        self.edgelist = list(self.graph.edges)

        self.set_lexicographic()

        self.set_faces(faces, no_faces=no_faces, verbose=verbose)

        self.create_matrices()

    def set_faces(self, faces=None, no_faces=False, verbose=True):
        """Set faces from list of triangles if provided, or all triangles."""
        if no_faces:
            self.faces = None
            self.n_faces = 0
        elif faces == None:
            all_cliques = nx.enumerate_all_cliques(self.graph)
            self.faces = [clique for clique in all_cliques if len(clique) == 3]
            self.n_faces = len(self.faces)
        else:
            self.faces = faces
            self.n_faces = len(self.faces)
            
        if verbose:
            print(f'We created {self.n_faces} faces')

    def set_lexicographic(self):
        """Set orientation of edges in lexicographic order."""
        for ei, e in enumerate(self.edgelist):
            self.edgelist[ei] = tuple(sorted(e))

    def flip_edge_orientation(self, edge_indices):
        """Flip the orientation of an edge."""
        if not isinstance(edge_indices, list):
            edge_indices = [edge_indices]
        for edge_index in edge_indices:
            self.edgelist[edge_index] = self.edgelist[edge_index][::-1]
        self.create_matrices()

    def create_matrices(self):
        """Create all needed weights and incidence matrices."""
        self.create_node_incidence_matrix()
        self.create_edge_incidence_matrix()

        self.create_node_weights_matrix()
        self.create_edge_weights_matrix()
        self.create_face_weights_matrix()

    def create_node_weights_matrix(self):
        """Create node weight matrix."""
        try:
            node_weights = [self.graph.nodes[u]["weight"] for u in self.graph]
        except:
            node_weights = np.ones(self.n_nodes)
        self.node_weights_matrix = sc.sparse.spdiags(
            node_weights, 0, self.n_nodes, self.n_nodes
        )

    def create_edge_weights_matrix(self):
        """Create edge weight matrix."""
        try:
            edge_weights = [self.graph[u][v]["weight"] for u, v in self.graph.edges]
        except:
            edge_weights = np.ones(self.n_edges)
        self.edge_weights_matrix = sc.sparse.spdiags(
            edge_weights, 0, self.n_edges, self.n_edges
        )

    def create_face_weights_matrix(self):
        """Create face weight matrix."""
        if self.faces is None:
            self.face_weights_matrix = None
        else:
            face_weights = np.ones(self.n_faces)
            self.face_weights_matrix = sc.sparse.spdiags(
                face_weights, 0, self.n_faces, self.n_faces
            )

    def create_node_incidence_matrix(self):
        """Create node incidence matrix."""
        self.node_incidence_matrix = nx.incidence_matrix(
            self.graph, edgelist=self.edgelist, oriented=True
        ).T

    def create_edge_incidence_matrix(self):
        """Create edge incidence matrix."""
        if self.faces == None:
            self.edge_incidence_matrix = None
        else:
            self.edge_incidence_matrix = sc.sparse.lil_matrix(
                (self.n_faces, self.n_edges)
            )
            for face_index, face in enumerate(self.faces):
                for i in range(3):
                    edge = tuple(np.roll(face, i)[:2])
                    edge_rev = tuple(np.roll(face, i)[1::-1])
                    if edge in self.edgelist:
                        edge_index = self.edgelist.index(edge)
                        self.edge_incidence_matrix[face_index, edge_index] = 1.0
                    elif edge_rev in self.edgelist:
                        edge_index = self.edgelist.index(edge_rev)
                        self.edge_incidence_matrix[face_index, edge_index] = -1.0
                    else:
                        raise Exception("The face is not a triangle in the graph")
