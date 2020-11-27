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

        self._B0 = None
        self._B1 = None
        self._W0 = None
        self._W1 = None
        self._W2 = None
        self._L0 = None
        self._L1 = None

        self._V = None
        self._lifted_B0 = None
        self._lifted_B0_p = None
        self._lifted_B0_n = None
        self._lifted_B1 = None
        self._lifted_B1_p = None
        self._lifted_B1_n = None
        self._lifted_L0 = None
        self._lifted_L1 = None

        self.set_lexicographic()
        self.set_faces(faces, no_faces=no_faces, verbose=verbose)

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
            print(f"We created {self.n_faces} faces")

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
        self._edge_incidence_matrix = None

    @property
    def W0(self):
        """Create node weight matrix."""
        if self._W0 is None:
            try:
                node_weights = [self.graph.nodes[u]["weight"] for u in self.graph]
            except:
                node_weights = np.ones(self.n_nodes)
            self._W0 = sc.sparse.spdiags(node_weights, 0, self.n_nodes, self.n_nodes)
        return self._W0

    @property
    def W1(self):
        """Create edge weight matrix."""
        if self._W1 is None:
            try:
                edge_weights = [self.graph[u][v]["weight"] for u, v in self.graph.edges]
            except:
                edge_weights = np.ones(self.n_edges)
            self._W1 = sc.sparse.spdiags(edge_weights, 0, self.n_edges, self.n_edges)
        return self._W1

    @property
    def W2(self):
        """Create face weight matrix."""
        if self._W2 is None:
            if self.faces is not None:
                face_weights = np.ones(self.n_faces)
                self._W2 = sc.sparse.spdiags(
                    face_weights, 0, self.n_faces, self.n_faces
                )
        return self._W2

    @property
    def B0(self):
        """Create node incidence matrix."""
        if self._B0 is None:
            self._B0 = nx.incidence_matrix(
                self.graph, edgelist=self.edgelist, oriented=True
            ).T
        return self._B0

    @property
    def B1(self):
        """Create edge incidence matrix."""
        if self._B1 is None:
            if self.faces is not None:
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
    def L0(self):
        """Compute the node laplacian."""
        if self._L0 is None:
            W1_inv = self.W1.copy()
            W1_inv.data = 1.0 / W1_inv.data
            self._L0 = self.W0.dot(self.B0.T).dot(W1_inv).dot(self.B0)
        return self._L0

    @property
    def L1(self):
        """Compute the edge laplacian."""
        if self._L1 is None:
            W1_inv = self.W1.copy()
            W1_inv.data = 1.0 / W1_inv.data
            self._L1 = self.B0.dot(self.W0).dot(self.B0.T).dot(W1_inv)

            if self.W2 is not None:
                W2_inv = self.W2.copy()
                W2_inv.data = 1.0 / W2_inv.data
                self._L1 += self.W1.dot(self.B1.T).dot(W2_inv).dot(self.B1)
        return self._L1

    @property
    def V(self):
        if self._V is None:
            self._V = sc.sparse.csr_matrix(
                np.concatenate((np.eye(self.n_edges), -np.eye(self.n_edges)), axis=0)
            )
        return self._V

    @property
    def lifted_B0(self):
        """Create lifted version of incidence matrices."""
        if self._lifted_B0 is None:
            self._liftted_B0 = self.V.dot(self.B0)
        return self._liftted_B0

    @property
    def lifted_B0_p(self):
        """Create the positive part lifted version of incidence matrices."""
        if self._lifted_B0_p is None:
            temp=self.lifted_B0.copy()
            temp[temp<0] = 0
            self._lifted_B0_p = temp
        return self._lifted_B0_p
    
    @property
    def lifted_B0_n(self):
        """Create the negative part lifted version of incidence matrices."""
        if self._lifted_B0_n is None:
            temp=self.lifted_B0.copy()
            temp[temp>0] = 0
            self._lifted_B0_n = np.negative(temp)
        return self._lifted_B0_n
    
    @property
    def lifted_B1(self):
        """Create lifted version of incidence matrices."""
        if self._lifted_B1 is None:
            self._liftted_B1 = self.B1.dot(self.V.T)
        return self._liftted_B1
    
    @property
    def lifted_B1_p(self):
        """Create the positive part lifted version of incidence matrices."""
        if self._lifted_B1_p is None:
            temp=self.lifted_B1.copy()
            temp[temp<0] = 0
            self._lifted_B1_p = temp
        return self._lifted_B1_p
    
    @property
    def lifted_B1_n(self):
        """Create the negative part lifted version of incidence matrices."""
        if self._lifted_B1_n is None:
            temp=self.lifted_B1.copy()
            temp[temp>0] = 0
            self._lifted_B1_n = np.negative(temp)
        return self._lifted_B1_n

    @property
    def lifted_L0(self):
        """Get lifted node laplacian."""
        if self._lifted_L0 is None:
            lifted_B0_pos = self.lifted_B0.copy()
            lifted_B0_pos[lifted_B0_pos < 0] = 0
            self._lifted_L0 = lifted_B0_pos.T.dot(self.lifted_B0)
        return self._lifted_L0

    @property
    def lifted_L1(self):
        """Get lifted edge laplacian."""
        if self._lifted_L1 is None:
            lifted_B0_pos = self.lifted_B0.copy()
            lifted_B0_pos[lifted_B0_pos < 0] = 0
            self._lifted_L1 = self.lifted_B0.dot(lifted_B0_pos.T)
#            print(self.W2)
            if self.W2 is not None:
                lifted_B1_pos = self.lifted_B1.copy()
                lifted_B1_pos[lifted_B1_pos < 0] = 0
                self._lifted_L1 += lifted_B1_pos.T.dot(self.lifted_B1)
        return self._lifted_L1

    def remove_zero_weight_edges_faces(self, return_idx=False):
        """This iis broken!"""
        B0 = self.B0.toarray()
        W0 = self.W0.toarray()
        B1 = self.B1.toarray()
        W1 = self.W1.toarray()
        W2 = self.W2.toarray()

        zero_weight_edges = W1.any(axis=1)
        zero_weight_faces = W2.any(axis=1)

        # remove edges from node incidence matrix
        B0 = np.delete(B0, np.where(~zero_weight_edges), axis=0)

        # remove edges from edge weight matrix
        W1 = np.delete(W1, np.where(~zero_weight_edges), axis=0)
        W1 = np.delete(W1, np.where(~zero_weight_edges), axis=1)

        # remove faces from edge incidence matrix
        B1 = np.delete(B1, np.where(~zero_weight_faces), axis=0)
        B1 = np.delete(B1, np.where(~zero_weight_edges), axis=1)

        # remove edges from edge weight matrix
        W2 = np.delete(W2, np.where(~zero_weight_faces), axis=0)
        W2 = np.delete(W2, np.where(~zero_weight_faces), axis=1)

        self._B0 = sc.sparse.lil_matrix(B0)
        self._W0 = sc.sparse.lil_matrix(W0)
        self._B1 = sc.sparse.lil_matrix(B1)
        self._W1= sc.sparse.spdiags(
            np.diagonal(W1), 0, W1.shape[0], W1.shape[0]
        )
        self._W2 = sc.sparse.spdiags(
            np.diagonal(W2), 0, W2.shape[0], W2.shape[0]
        )

        self.n_edges = W1.shape[0]
        self.n_faces = W2.shape[0]

        self.graph = nx.Graph(self.graph)

        # remove edges from nx graph
        for edge_id in np.where(~zero_weight_edges)[0]:
            edge = self.edgelist[edge_id]
            self.graph.remove_edge(edge[0], edge[1])

        return zero_weight_edges, zero_weight_faces
