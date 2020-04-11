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
        self.edgelist = list(self.graph.edges)
        self.set_lexicographic()

        self.n_nodes = len(self.graph)
        self.n_edges = len(self.graph.edges)
        # self.n_faces = len(self.faces)

        self.create_matrices()

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
        self.degree = np.array([len(self.graph[u]) for u in self.graph])

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
            # Edge incidence matrix
            A = nx.to_numpy_matrix(self.graph)
            Nn = A.shape[0]
            Ne = int(np.sum(A) / 2)

            e = np.zeros((Ne, 2))
            count = 0
            for i in range(Nn):
                for j in range(i + 1, Nn):
                    if A[i, j] > 0:
                        e[count, 0] = i
                        e[count, 1] = j
                        count += 1
            # print("edges")
            # print(e)

            Nf = 0
            for i in range(Nn):
                for j in range(i + 1, Nn):
                    for k in range(j + 1, Nn):
                        subA = A[np.ix_([i, j, k], [i, j, k])]
                        if np.sum(subA) == 6:
                            Nf += 1
            f = np.zeros((Nf, 3))
            count = 0
            for i in range(Nn):
                for j in range(i + 1, Nn):
                    for k in range(j + 1, Nn):
                        subA = A[np.ix_([i, j, k], [i, j, k])]
                        if np.sum(subA) == 6:
                            f[count, 0] = i
                            f[count, 1] = j
                            f[count, 2] = k
                            count += 1
            # print("faces")
            # print(f)
            II = np.zeros((Nf, Ne))
            for i in range(f.shape[0]):
                for j in [0, -1, -2]:
                    temp = np.roll(f[i, :], j)
                    temp = temp[0:2]
                    for k in range(e.shape[0]):
                        # print e[k,:],temp
                        if ((e[k, :] == temp).all()) or (
                            (e[k, :] == np.roll(temp, 1)).all()
                        ):
                            Irow = k
                    if temp[0] < temp[1]:
                        II[i, Irow] = 1
                    else:
                        II[i, Irow] = -1
            # print II
            #        ntrie=np.sum(II,1)
            self.edge_incidence_matrix = II

        #        return I,II#,ntrie, e#, len(ntrie)
