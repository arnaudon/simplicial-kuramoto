"""Representation of simplicial complex"""
import networkx as nx
import numpy as np
import scipy as sc


def pos(matrix):
    """Return positive part of matrix."""
    _matrix = matrix.copy()
    _matrix[_matrix < 0] = 0
    return _matrix


def neg(matrix):
    """Return negative part of matrix."""
    _matrix = matrix.copy()
    _matrix[_matrix > 0] = 0
    return _matrix


class SimplicialComplex:
    """Class representing a simplicial complex."""

    def __init__(self, graph=None, faces=None, no_faces=False, face_weights=None):
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

        self._B0 = None
        self._N0 = None
        self._N0s = None
        self._B1 = None
        self._N1 = None
        self._N1s = None
        self._W0 = None
        self._W1 = None
        self._W2 = None
        self._L0 = None
        self._L1 = None

        self._V1 = None
        self._V2 = None

        self._lifted_N0 = None
        self._lifted_N0sn = None
        self._lifted_N1 = None
        self._lifted_N1sn = None

        self.set_lexicographic()
        self.no_faces = no_faces
        self.set_faces(faces)
        if face_weights is None:
            self.face_weights = np.ones(self.n_faces)
        else:
            self.face_weights = face_weights

    def set_faces(self, faces=None):
        """Set faces from list of triangles if provided, or all triangles."""
        if self.no_faces:
            self.faces = None
            self.n_faces = 0
        elif faces is None:
            all_cliques = nx.enumerate_all_cliques(self.graph)
            self.faces = [clique for clique in all_cliques if len(clique) == 3]
            self.n_faces = len(self.faces)
        else:
            self.faces = faces
            self.n_faces = len(self.faces)

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

        self._B0 = None
        self._N0 = None
        self._N0s = None
        self._B1 = None
        self._N1 = None
        self._N1s = None
        self._W0 = None
        self._W1 = None
        self._W2 = None
        self._L0 = None
        self._L1 = None
        self._V2 = None
        self._lifted_N1 = None
        self._lifted_N1sn = None

        self.set_faces(self.faces)

    @property
    def W0(self):
        """Create node weight matrix."""
        if self._W0 is None:
            self._W0 = sc.sparse.spdiags(
                [self.graph.nodes[u].get("weight", 1.0) for u in self.graph],
                0,
                self.n_nodes,
                self.n_nodes,
            )
        return self._W0

    @property
    def W1(self):
        """Create edge weight matrix."""
        if self._W1 is None:
            self._W1 = sc.sparse.spdiags(
                [self.graph[u][v].get("weight", 1.0) for u, v in self.graph.edges],
                0,
                self.n_edges,
                self.n_edges,
            )
        return self._W1

    @property
    def W2(self):
        """Create face weight matrix."""
        if self._W2 is None and self.faces is not None:
            self._W2 = sc.sparse.spdiags(self.face_weights, 0, self.n_faces, self.n_faces)
        return self._W2

    @property
    def B0(self):
        """Create node incidence matrix."""
        if self._B0 is None:
            self._B0 = nx.incidence_matrix(self.graph, edgelist=self.edgelist, oriented=True).T
        return self._B0

    @property
    def N0(self):
        """Create weighted node incidence matrix."""
        if self._N0 is None:
            self._N0 = self.B0
        return self._N0

    @property
    def N0s(self):
        """Create conjugate weighted node incidence matrix."""
        if self._N0s is None:
            W1_inv = self.W1.copy()
            W1_inv.data = 1.0 / W1_inv.data
            self._N0s = self.W0.dot(self.B0.T).dot(W1_inv)
        return self._N0s

    @property
    def B1(self):
        """Create edge incidence matrix."""
        if self._B1 is None and self.faces is not None:
            self._B1 = sc.sparse.lil_matrix((self.n_faces, self.n_edges))
            for face_index, face in enumerate(self.faces):
                for i in range(3):
                    edge = tuple(np.roll(face, i)[:2])
                    edge_rev = tuple(np.roll(face, i)[1::-1])
                    if edge in self.edgelist:
                        edge_index = self.edgelist.index(edge)
                        self._B1[face_index, edge_index] = 1.0
                    elif edge_rev in self.edgelist:
                        edge_index = self.edgelist.index(edge_rev)
                        self._B1[face_index, edge_index] = -1.0
                    else:
                        raise Exception("The face is not a triangle in the graph")
        return self._B1

    @property
    def N1s(self):
        """Create conjugate weighted node incidence matrix."""
        if self._N1s is None:
            W2_inv = self.W2.copy()
            W2_inv.data = 1.0 / W2_inv.data
            self._N1s = self.W1.dot(self.B1.T).dot(W2_inv)
        return self._N1s

    @property
    def N1(self):
        """Create conjugate weighted edge incidence matrix."""
        if self._N1 is None:
            self._N1 = self.B1
        return self._N1

    @property
    def L0(self):
        """Compute the node laplacian."""
        if self._L0 is None:
            self._L0 = self.N0s.dot(self.N0)
        return self._L0

    @property
    def L1(self):
        """Compute the edge laplacian."""
        if self._L1 is None:
            self._L1 = self.N0.dot(self.N0s)

            if self.W2 is not None:
                self._L1 += self.N1s.dot(self.N1)
        return self._L1

    @property
    def V1(self):
        """Lift operator on faces."""
        if self._V1 is None:
            self._V1 = sc.sparse.csr_matrix(
                np.concatenate((np.eye(self.n_edges), -np.eye(self.n_edges)), axis=0)
            )
        return self._V1

    @property
    def V2(self):
        """Lift operator on faces."""
        if self._V2 is None:
            self._V2 = sc.sparse.csr_matrix(
                np.concatenate((np.eye(self.n_faces), -np.eye(self.n_faces)), axis=0)
            )
        return self._V2

    @property
    def lifted_N0(self):
        """Create lifted version of incidence matrices."""
        if self._lifted_N0 is None:
            self._lifted_N0 = self.V1.dot(self.N0)
        return self._lifted_N0

    @property
    def lifted_N0sn(self):
        """Create lifted version of incidence matrices."""
        if self._lifted_N0sn is None:
            self._lifted_N0sn = neg(self.N0s.dot(self.V1.T))
        return self._lifted_N0sn

    @property
    def lifted_N1(self):
        """Create lifted version of incidence matrices."""
        if self._lifted_N1 is None:
            self._lifted_N1 = self.V2.dot(self.N1)
        return self._lifted_N1

    @property
    def lifted_N1sn(self):
        """Create lifted version of incidence matrices."""
        if self._lifted_N1sn is None:
            self._lifted_N1sn = neg(self.N1s.dot(self.V2.T))
        return self._lifted_N1sn
