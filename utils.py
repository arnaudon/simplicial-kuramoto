import numpy as np
import pylab as plt
import networkx as nx
import scipy.integrate as integ


def compute_eig_projection(G):
    """
    Given a networkx graph, compute the B, eigenvalues w and eigenvectors v
    """
    
    B = nx.incidence_matrix(G, oriented = True).toarray().T
    B = np.repeat(B, 2, axis = 0)
    
    for i in range(0, B.shape[0], 2):
        B[i,:] = -B[i,:]
    
    Bplus = 0.5*(np.abs(B) + B)

    L = B.T.dot(B)#nx.laplacian_matrix(G).toarray()

    w, v = np.linalg.eig(L) #eigenvalues/eigenvectors
    
    #sort them by increasing w
    w_sort = np.argsort(w)
    w = np.real(w[w_sort])
    v = np.real(v[:,w_sort])
    
    return B, Bplus, v, w


###############################
# integration of full kuramoto
################################

def kuramoto_full_theta(t, theta, B, Bplus, alpha, a, omega_0, degree):
    #kuramoto ODE in physical space
    
    return omega_0 - a*(1/degree) * Bplus.T.dot( np.sin( B.dot(theta) + alpha) )


def kuramoto_full_gamma(t, gamma, B, Bplus, v, alpha, a, omega_0, degree):
    #kuramoto ODE in spectral space

    return omega_0.dot(v) - a*(1/degree) * (Bplus.T.dot( np.sin( B.dot( gamma.dot(v.T) +alpha) ) ) ).dot(v)

def integrate_kuramoto_full_theta(B, Bplus, theta_0, t_max, n_t, alpha, a, omega_0):
    #integrate Kuramoto ODE in physical space
    
    degree=np.absolute(B).sum(0)
    kuramoto_integ = lambda t, theta: kuramoto_full_theta(t, theta, B, Bplus, alpha, a, omega_0, degree)
    
    return integ.solve_ivp(kuramoto_integ, [0, t_max], theta_0, t_eval = np.linspace(0, t_max, n_t), method='LSODA', rtol = 1.49012e-8, atol = 1.49012e-8)

def integrate_kuramoto_full_gamma(B, Bplus, v, gamma_0, t_max, n_t, alpha, a, omega_0):
    #integrate Kuramoto ODE in spectral space
    
    degree=np.absolute(B).sum(0)
    kuramoto_integ = lambda t, theta: kuramoto_full_gamma(t, theta, B, Bplus, v, alpha, a, omega_0, degree)
    
    return  integ.solve_ivp(kuramoto_integ, [0, t_max], gamma_0, t_eval = np.linspace(0, t_max, n_t), method='LSODA', rtol = 1.49012e-8, atol = 1.49012e-8)




##########################
## graph generation
########################

def Modules_full(Nc,Nn,Nie,rando=True):
    """"
    Produces a modular network with Nc clique modules of size Nn and all connected by Nie edges, in a linear way
        Nc: number of modules
        Nn: number of nodes per module
        Nie: number of edges between modules (added linearly), has to be smaller than Nn(Nn-1)/2
    """
    
    G=nx.Graph()
    G.add_nodes_from(np.linspace(1,Nc*Nn,Nc*Nn).astype(int).tolist())
    
    # fully connected modules
    for i in range(Nc):
        for j in range(1,Nn+1):
             for k in range(j+1,Nn+1):
                    G.add_edge((i*Nn)+j,(i*Nn)+k)
    
    if rando:
        for i in range(Nc):
            for j in range(i+1,Nc):
                source=np.random.randint(i*Nn+1,(i+1)*Nn,Nie).tolist()
                sink=np.random.randint(j*(Nn+1)+1,(j+1)*Nn,Nie).tolist()
                for e in range(Nie):
                    G.add_edge(source[e],sink[e])
                    
    else:
        Neig = np.linspace(1,Nn,Nn).astype(int)
        if Nie>0:
            nr,bonus=np.divmod(Nie,Nn)
            for c1 in range(Nc):
                for c2 in range(c1+1,Nc):
                    for i in range(nr):
                        for j in range(Nn):
                            G.add_edge(Neig.tolist()[j]+c1*Nn,np.roll(Neig,-i).tolist()[j]+c2*Nn)
                    for j in range(bonus):
                        G.add_edge(Neig.tolist()[j]+c1*Nn,np.roll(Neig,-(nr)).tolist()[j]+c2*Nn)

    return G

#######################################
## compute order and shanahan indices
#######################################

def order_parameter(ts, Nc, Nn):
    """
    Computes the order parameter for the whole system and each community (assuming there are more time points than nodes)
    need to have it more flexible based on node indices
    """
    
    d1, d2 = ts.shape
    if d1 > d2:
        op = np.zeros((Nc+1,d1))
        op[0,:] = np.absolute(np.exp(1j*ts).sum(1))/d2
        for i in range(Nc):
            op[i+1,:] = np.absolute(np.exp(1j*ts[:,i*Nn:(i+1)*Nn]).sum(1))/Nn
    else:
        op=np.zeros((Nc+1,d2))
        op[0,:] = np.absolute(np.exp(1j*ts).sum(0))/d1
        for i in range(Nc):
            op[i+1,:] = np.absolute(np.exp(1j*ts[i*Nn:(i+1)*Nn,:]).sum(0))/Nn
            
    return op

def Shanahan_indices(op):
    """
    compute the two Shanahan indices
    
        l is the average across communities of the variance of the order parameter within communities ("global" metastability)
        chi is the avarage across time of the variance of the order parameter across communities at time t (Chimeraness of the system)
        op should have dimensions (number of communities+1,time), the plus one is for global order parameter on the first row
    """
    
    l = np.var(op[1:op.shape[1]], axis=1).mean()
    chi = np.var(op[1:op.shape[1]], axis=0).mean()
    
    return l, chi

#######################################
#compute the Delta tensors naively
######################################

def Delta_1(Bv, Bplusv):
    return Bplusv.sum(0)

def Delta_2(Bv, Bplusv):

    return (Bplusv[:, :, np.newaxis]*Bv[:, np.newaxis,: ]).sum(0)

def Delta_3(Bv, Bplusv):

    return (Bplusv[:, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :]).sum(0)

def Delta_4(Bv, Bplusv):

    return (Bplusv[:, :, np.newaxis, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, np.newaxis, :]).sum(0)


