import numpy as np
import pylab as plt
import networkx as nx
import scipy.integrate as integ


def compute_eig_projection(G ):
    """
    Given a networkx graph, compute the B, eigenvalues w and eigenvectors v
    """
    
    B = nx.incidence_matrix(G,oriented = True).toarray()
    L = nx.laplacian_matrix(G).toarray()

    w, v = np.linalg.eig(L) #eigenvalues/eigenvectors
    
    #sort them by increasing w
    w_sort = np.argsort(w)
    w = w[w_sort]
    v = v[:,w_sort]
    
    return B, v, w


#compute the Delta tensors naively
def Delta_1(Bv):
    return np.asarray(Bv.sum(0)).flatten()

def Delta_2(Bv):
    return (Bv[:, :, np.newaxis]*Bv[:, np.newaxis,: ]).sum(0)

def Delta_3(Bv):
    return (Bv[:, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :]).sum(0)

def Delta_4(Bv):
    return (Bv[:, :, np.newaxis, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, np.newaxis, :]).sum(0)


def kuramoto_full_theta(t, theta, B):
    return -B.dot(np.sin(B.T.dot(theta)))

def kuramoto_full_gamma(t, gamma, B, v):
    return -(B.dot(np.sin(B.T.dot(gamma.dot(v.T))))).dot(v)