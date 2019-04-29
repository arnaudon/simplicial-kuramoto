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
    
    L = Bplus.T.dot(B)#nx.laplacian_matrix(G).toarray()

    w, v = np.linalg.eigh(L) #eigenvalues/eigenvectors
    
    #sort them by increasing w
    w_sort = np.argsort(w)
    w = np.real(w[w_sort])
    v = np.real(v[:,w_sort])
    
    return B, Bplus, v, w

def compute_eig_projection_weighted(G,mu=1,nu=1,Nn=1):
    """
    Given a networkx graph, compute the B, eigenvalues w and eigenvectors v
    """

    B = nx.incidence_matrix(G, oriented = True).toarray().T
    B = np.repeat(B, 2, axis = 0)
    
    for i in range(0, B.shape[0], 2):
        B[i,:] = -B[i,:]
    
    Bplus = 0.5*(np.abs(B) + B)
    
    # works only for sbm type with clusters of the same size
    W_e=np.ones(B.shape[0])
    for i in range(B.shape[0]):
        if (np.floor_divide(np.nonzero(B[i,:])[0][0],Nn)==np.floor_divide(np.nonzero(B[i,:])[0][1],Nn)):
            W_e[i]=mu
        else:
            W_e[i]=nu
    W_e=np.diag(W_e)
    
    L = Bplus.T.dot(W_e.dot(B))#nx.laplacian_matrix(G).toarray()

    w, v = np.linalg.eigh(L) #eigenvalues/eigenvectors
    
    #sort them by increasing w
    w_sort = np.argsort(w)
    w = np.real(w[w_sort])
    v = np.real(v[:,w_sort])
    
    return B, Bplus, W_e, v, w


###############################
# integration of full kuramoto
###############################

def kuramoto_full_theta(t, theta, B, Bplus, alpha, a, omega_0, degree):
    #kuramoto ODE in physical space
    
    return omega_0 - a/degree * Bplus.T.dot( np.sin( B.dot(theta) + alpha) )

def kuramoto_full_theta_weighted(t, theta, B, Bplus, alpha, a, omega_0, degree, W_e):
    #kuramoto ODE in physical space
    
    return omega_0 - a/degree * Bplus.T.dot(W_e.dot( np.sin( B.dot(theta) + alpha) ))

def kuramoto_full_gamma(t, gamma, B, Bplus, v, alpha, a, omega_0, degree):
    #kuramoto ODE in spectral space
    return omega_0.dot(v) - a/degree * (Bplus.T.dot( np.sin( B.dot(gamma.dot(v.T)) + alpha) ) ).dot(v)

def integrate_kuramoto_full_theta(B, Bplus, theta_0, t_max, n_t, alpha, a, omega_0):
    #integrate Kuramoto ODE in physical space
    
    degree = np.absolute(Bplus).sum(0)
    kuramoto_integ = lambda t, theta: kuramoto_full_theta(t, theta, B, Bplus, alpha, a, omega_0, degree)
    
    return integ.solve_ivp(kuramoto_integ, [0, t_max], theta_0, t_eval = np.linspace(0, t_max, n_t), method='LSODA', rtol = 1.49012e-8, atol = 1.49012e-8)

def integrate_kuramoto_full_theta_weighted(B, Bplus, theta_0, t_max, n_t, alpha, a, omega_0, W_e):
    #integrate Kuramoto ODE in physical space
    
    degree = np.diag(Bplus.T.dot(W_e.dot(B)))
    kuramoto_integ = lambda t, theta: kuramoto_full_theta_weighted(t, theta, B, Bplus, alpha, a, omega_0, degree, W_e)
    
    return integ.solve_ivp(kuramoto_integ, [0, t_max], theta_0, t_eval = np.linspace(0, t_max, n_t), method='LSODA', rtol = 1.49012e-8, atol = 1.49012e-8)

def integrate_kuramoto_full_gamma(B, Bplus, v, gamma_0, t_max, n_t, alpha, a, omega_0):
    #integrate Kuramoto ODE in spectral space
    
    degree=np.absolute(Bplus).sum(0)
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

######################################
# approximate Kuramoto
######################################


def Delta_1(Bv, Bplusv):
    return Bplusv.sum(0)

def Delta_2(Bv, Bplusv):
    return (Bplusv[:, :, np.newaxis]*Bv[:, np.newaxis,: ]).sum(0)

def Delta_3(Bv, Bplusv):
    return (Bplusv[:, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :]).sum(0)

def Delta_4(Bv, Bplusv):
    return (Bplusv[:, :, np.newaxis, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, np.newaxis, :]).sum(0)

def Delta_ids(Bv, Bplusv, w, th_3, th_4):
    """
    find the indices of above threshold elements of D_3 and D_4
    """
    
    w_0 = w.copy()
    w_0[0] = 1

    D3 = Delta_3(Bv, Bplusv)/w_0[:,np.newaxis,np.newaxis]
    D4 = Delta_4(Bv, Bplusv)/w_0[:,np.newaxis,np.newaxis,np.newaxis]

    D3_id = np.argwhere(abs(D3) > th_3)
    D4_id = np.argwhere(abs(D4) > th_4)
    
    return D3_id, D4_id

def kuramoto_approx(t, gamma, D_1, D_3, D_4, w,  alpha, a, omega_0, degree, D3_id, D4_id):
    """
    kuramoto ODE in spectral space
    """

    #low order terms
    f = -(alpha - alpha**3/6.)*D_1 - (1. - alpha**2/2.)*gamma*w

    #cubic term
    for i in D4_id:
        f[i[0]] += 1./6.*D_4[i[0],i[1],i[2],i[3]]*gamma[i[1]]*gamma[i[2]]*gamma[i[3]]
        
    #quadratic term
    for i in D3_id:
        f[i[0]] += alpha/2.*D_3[i[0],i[1],i[2]]*gamma[i[1]]*gamma[i[2]]

    return f/degree

def integrate_kuramoto_approx(B, Bplus, v, w, gamma_0, t_max, n_t, alpha, a, omega_0, th_3, th_4):
    """
    integrate approximated Kuramoto ODE in spectral space
    """
    
    degree = np.absolute(Bplus).sum(0)
    
    Bv = np.array(B.dot(v)) #edges by modes
    Bplusv = np.array(Bplus.dot(v)) #edges by modes
    
    D1 = Delta_1(Bv, Bplusv)
    D3 = Delta_3(Bv, Bplusv)
    D4 = Delta_4(Bv, Bplusv)
    
    D3_id, D4_id = Delta_ids(Bv, Bplusv, w, th_3, th_4)
    
    print('Using ', len(D3_id),'elements of Delta_3')
    print('Using ', len(D4_id),'elements of Delta_4')
    
    kuramoto_integ = lambda t, gamma: kuramoto_approx(t, gamma, D1, D3, D4, w, alpha, a, omega_0, degree, D3_id, D4_id)
    
    sol = integ.solve_ivp(kuramoto_integ, [0, t_max], gamma_0, t_eval = np.linspace(0, t_max, n_t), method='LSODA', rtol = 1.49012e-8, atol = 1.49012e-8)
    
    return sol, len(D3_id), len(D4_id)

######################################
# approximate Kuramoto weighted
######################################

def Delta_1_weighted(Bv, Bplusv, W_e):
    return (np.diag(W_e)[:,np.newaxis]*Bplusv).sum(0) # W_e.dot(Bplusv).sum(0) and also = v.T.dot(degree)

def Delta_2_weighted(Bv, Bplusv, W_e):
     return (np.diag(W_e)[:,np.newaxis,np.newaxis]*(Bplusv[:, :, np.newaxis]*Bv[:, np.newaxis,: ])).sum(0) # = w

def Delta_3_weighted(Bv, Bplusv, W_e):
    return (np.diag(W_e)[:,np.newaxis,np.newaxis,np.newaxis]*Bplusv[:, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :]).sum(0)

def Delta_4_weighted(Bv, Bplusv, W_e):
    return (np.diag(W_e)[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*Bplusv[:, :, np.newaxis, np.newaxis, np.newaxis]*Bv[:, np.newaxis, :, np.newaxis, np.newaxis]*Bv[:, np.newaxis, np.newaxis, :, np.newaxis]*Bv[:, np.newaxis, np.newaxis, np.newaxis, :]).sum(0)

def Delta_ids_weighted(Bv, Bplusv, w, th_3, th_4, W_e):
    """
    find the indices of above threshold elements of D_3 and D_4
    """
    
    w_0 = w.copy()
    w_0[0] = 1

    D3_weighted = Delta_3_weighted(Bv, Bplusv, W_e)/w_0[:,np.newaxis,np.newaxis]
    D4_weighted = Delta_4_weighted(Bv, Bplusv, W_e)/w_0[:,np.newaxis,np.newaxis,np.newaxis]

    D3_id_weighted = np.argwhere(abs(D3_weighted) > th_3)
    D4_id_weighted = np.argwhere(abs(D4_weighted) > th_4)
    
    return D3_id_weighted, D4_id_weighted

def kuramoto_approx_weighted(t, gamma, D_1_weighted, D_3_weighted, D_4_weighted, w,  alpha, a, omega_0, degree, D3_id_weighted, D4_id_weighted):
    """
    kuramoto ODE in spectral space
    """

    #low order terms
    f = -(alpha - alpha**3/6.)*D_1_weighted - (1. - alpha**2/2.)*gamma*w

    #cubic term
    for i in D4_id_weighted:
        f[i[0]] += 1./6.*D_4_weighted[i[0],i[1],i[2],i[3]]*gamma[i[1]]*gamma[i[2]]*gamma[i[3]]
        
    #quadratic term
    for i in D3_id_weighted:
        f[i[0]] += alpha/2.*D_3_weighted[i[0],i[1],i[2]]*gamma[i[1]]*gamma[i[2]]

    return f/degree

def integrate_kuramoto_approx_weighted(B, Bplus, v, w, gamma_0, t_max, n_t, alpha, a, omega_0, th_3, th_4, W_e):
    """
    integrate approximated Kuramoto ODE in spectral space
    """
    
    degree = np.diag(Bplus.T.dot(W_e.dot(B)))
    
    Bv = np.array(B.dot(v)) #edges by modes
    Bplusv = np.array(Bplus.dot(v)) #edges by modes
    
    D1_weighted = Delta_1_weighted(Bv, Bplusv, W_e)
    D3_weighted = Delta_3_weighted(Bv, Bplusv, W_e)
    D4_weighted = Delta_4_weighted(Bv, Bplusv, W_e)
    
    D3_id_weighted, D4_id_weighted = Delta_ids_weighted(Bv, Bplusv, w, th_3, th_4, W_e)
    
    print('Using ', len(D3_id_weighted),'elements of Delta_3')
    print('Using ', len(D4_id_weighted),'elements of Delta_4')
    
    kuramoto_integ_weighted = lambda t, gamma: kuramoto_approx_weighted(t, gamma, D1_weighted, D3_weighted, D4_weighted, w, alpha, a, omega_0, degree, D3_id_weighted, D4_id_weighted)
    
    sol = integ.solve_ivp(kuramoto_integ_weighted, [0, t_max], gamma_0, t_eval = np.linspace(0, t_max, n_t), method='LSODA', rtol = 1.49012e-8, atol = 1.49012e-8)
    
    return sol, len(D3_id_weighted), len(D4_id_weighted)

def compute_error(sol_full, sol_approx):
    """
    compute relative error between two solutions
    """
    
    err = []
    for t in range(np.shape(sol_full.y)[1]):
        err.append(np.linalg.norm(sol_full.y[1:,t]-sol_approx.y[1:,t]) /np.linalg.norm(sol_full.y[:,t]))
        
    return err
