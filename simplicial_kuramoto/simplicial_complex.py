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

        # signed incidence matrices for triangles
        A=self.graph
        Nn=A.shape[0]
        Ne=int(np.sum(A)/2)
        #print Nn, Ne

        # this part creates a signed node incidence matrix, in lexicographic order. Not sure it is redundant with the one above,
        # the orientaiton initially chosen for the node incidence matrix implies the orientation on the edge indicence matrix
        e=np.zeros((Ne,2))
        count=0;
        for i in range(Nn):
            for j in range(i+1,Nn):
                if(A[i,j]>0):
                    e[count,0]=i
                    e[count,1]=j
                    count+=1
        print "edges"
        print e
        I=np.zeros((Ne,Nn))
        for i in range(Ne):
            I[i,int(e[i,0])]=1
            I[i,int(e[i,1])]=-1
        #print I

        # Edge incidence matrix
        Nf=0
        for i in range(Nn):
            for j in range(i+1,Nn):
                for k in range(j+1,Nn):
                    subA=A[np.ix_([i,j,k],[i,j,k])]
                    if(np.sum(subA)==6):
                        Nf+=1
        f=np.zeros((Nf,3))
        count=0
        for i in range(Nn):
            for j in range(i+1,Nn):
                for k in range(j+1,Nn):
                    subA=A[np.ix_([i,j,k],[i,j,k])]
                    if(np.sum(subA)==6):
                        f[count,0]=i
                        f[count,1]=j
                        f[count,2]=k
                        count+=1
        print "faces"
        print f
        II=np.zeros((Nf,Ne))
        for i in range(f.shape[0]):
            for j in [0,-1,-2]:
                temp=np.roll(f[i,:],j)
                temp=temp[0:2]
                for k in range(e.shape[0]):
                    #print e[k,:],temp
                    if(((e[k,:]==temp).all())or((e[k,:]==np.roll(temp,1)).all())):
                        Irow=k
                if(temp[0]<temp[1]):
                    II[i,Irow]=1
                else:
                    II[i,Irow]=-1
        #print II 
#        ntrie=np.sum(II,1)
        self.edge_incidence_matrix=II
#        return I,II#,ntrie, e#, len(ntrie)

    def flip_edge_orientation(self, edge_index):
        """Flip the orientation of an edge."""
        self.node_incidence_matrix[edge_index] *= -1
