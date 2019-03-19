import numpy as np
import pylab as plt
import networkx as nx
import scipy.integrate as integ


def compute_eig_projection(G):
    """
    Given a networkx graph, compute the B, eigenvalues w and eigenvectors v
    """
    
    B = nx.incidence_matrix(G,oriented = True).toarray().T
    L = nx.laplacian_matrix(G).toarray()

    w, v = np.linalg.eig(L) #eigenvalues/eigenvectors
    
    #sort them by increasing w
    w_sort = np.argsort(w)
    w = w[w_sort]
    v = v[:,w_sort]
    
    return B, v, w

def Modules_full(Nc,Nn,Nie):
    """
    Produces a modular network with Nc clique modules of size Nn and all connected by Nie edges, in a linear way
    """
    # Nc: number of modules
    # Nn: number of nodes per module
    # Nie: number of edges between modules (added linearly), has to be smaller than Nn(Nn-1)/2
    G=nx.Graph()
    G.add_nodes_from(np.linspace(1,Nc*Nn,Nc*Nn).astype(int).tolist())
    # fully connected modules
    for i in range(Nc):
        for j in range(1,Nn+1):
             for k in range(j+1,Nn+1):
                    G.add_edge((i*Nn)+j,(i*Nn)+k)
    
    Neig=np.linspace(1,Nn,Nn).astype(int)
    if(Nie>0):
        nr,bonus=np.divmod(Nie,Nn)
        for c1 in range(Nc):
            for c2 in range(c1+1,Nc):
                for i in range(nr):
                    for j in range(Nn):
                        G.add_edge(Neig.tolist()[j]+c1*Nn,np.roll(Neig,-i).tolist()[j]+c2*Nn)
                for j in range(bonus):
                    G.add_edge(Neig.tolist()[j]+c1*Nn,np.roll(Neig,-(nr)).tolist()[j]+c2*Nn)

    return G


def order_parameter(ts,Nc,Nn):
    # Computes the order parameter for the whole graph (assuming there are more time points than nodes)
    d1,d2=ts.shape
    print d1,d2
    if (d1>d2):
        op=np.zeros((Nc+1,d1))
        op[0,:]=np.absolute(np.exp(1j*ts).sum(1))/d2
        for i in range(Nc):
            op[i+1,:]=np.absolute(np.exp(1j*ts[:,i*Nn:(i+1)*Nn]).sum(1))/Nn
    else:
        op=np.zeros((Nc+1,d2))
        op[0,:]=np.absolute(np.exp(1j*ts).sum(0))/d1
        for i in range(Nc):
            print i*Nn,(i+1)*Nn-1,ts[i*Nn:(i+1)*Nn,:].shape
            op[i+1,:]=np.absolute(np.exp(1j*ts[i*Nn:(i+1)*Nn,:]).sum(0))/Nn
    return op

#compute the Delta tensors naively
def Delta_1(Bv):
    return np.asarray(Bv.sum(0)).flatten()

def Delta_2(Bv):
    return (Bv[:, :, np.newaxis]*Bv[:, np.newaxis,: ]).sum(0)

def Delta_3(Bv):
    return (Bv[:, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :]).sum(0)

def Delta_4(Bv):
    return (Bv[:, :, np.newaxis, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, np.newaxis, :]).sum(0)


#models
def kuramoto_full_theta(t, theta, B):
    return -B.T.dot(np.sin(B.dot(theta)))

def kuramoto_full_gamma(t, gamma, B, v):
    return -(B.T.dot(np.sin(B.dot(gamma.dot(v.T))))).dot(v)

def kuramoto_full_theta_alpha(t, theta, B, alpha, a, omega_0, degree):
    return omega_0-a*(1/degree)*np.dot(B.T,np.sin(np.dot(B,theta)-alpha*np.ones(B.shape[0])))


def integrate_kuramoto_full_theta(B, theta_0, t_max, n_t):
    
    return integ.solve_ivp(lambda t, theta: kuramoto_full_theta(t, theta, B), [0, t_max], theta_0, t_eval = np.linspace(0, t_max, n_t))

def integrate_kuramoto_full_gamma(B, v, gamma_0, t_max, n_t):
    
    return  integ.solve_ivp(lambda t, theta: kuramoto_full_gamma(t, theta, B, v), [0, t_max], gamma_0, t_eval = np.linspace(0, t_max, n_t))

def integrate_kuramoto_full_theta_alpha(B, theta_0, t_max, n_t, alpha, a, omega_0, degree):
    
    return integ.solve_ivp(lambda t, theta: kuramoto_full_theta_alpha(t, theta, B, alpha, a, omega_0, degree), [0, t_max], theta_0, t_eval = np.linspace(0, t_max, n_t))


