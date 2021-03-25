"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import null_space


# from utils import *


def plot_node_kuramoto(node_results):
    """Basic plot for node kuramoto."""
    plt.figure()
    plt.imshow(
        node_results.y,
        aspect="auto",
        cmap="twilight_shifted",
        extent=(node_results.t[0], node_results.t[-1], 0, len(node_results.y)),
    )
    plt.xlabel("time")
    plt.ylabel("mode id")
    plt.colorbar()


def plot_edge_kuramoto(edge_results):
    """Basic plot for edge kuramoto."""
    plt.figure()
    plt.imshow(
        np.round(edge_results.y + np.pi, 2) %  (2 * np.pi) - np.pi,
        #edge_results.y,
        aspect="auto",
        cmap="twilight_shifted",
        interpolation='nearest',
        extent=(edge_results.t[0], edge_results.t[-1], 0, len(edge_results.y)),
    )
    plt.title("Phases")
    plt.colorbar()
    plt.show()


def plot_flow(initial_phase, simplicial_complex, result, plotname="Test"):
    """ Some outputs (curl, div, etc ...) are only useful for understanding what is happening """
    times = result.t
    phase = result.y

    B0 = simplicial_complex.node_incidence_matrix
    B1 = simplicial_complex.edge_incidence_matrix

    # op=order_parameter(phase, 4, 1) # that is in the utils.py
    # plt.figure()
    # plt.title('Order parameter')
    # plt.plot(op[0,:])

    if B1 == None:
        print("theta_0: ", initial_phase)
        print("theta_final: ", phase[:, -1])
        print(
            "theta_final: ",
            np.mod(np.around(phase[:, -1], 10), np.around(2 * np.pi, 10)),
        )

        Div = np.mod(np.around(B0.T.dot(phase), 10), np.around(2 * np.pi, 10))
        print("Div: ", Div[:, -1])
        print("Curl: no curl")

        L1 = -B0.dot(B0.T)
        print("L1 theta: ", L1.dot(phase[:, -1]))
        print(
            "L1 theta: ",
            np.mod(np.around(L1.dot(phase[:, -1]), 10), np.around(2 * np.pi, 10)),
        )

        # w, v=eigs(L1)
        # ns_ind=np.where(w<10e-8)
        # print("dim(Ker(L1)): ", len(ns_ind)

        ns = null_space(
            L1.toarray()
        )  # not ideal, would be better in principle to stay in sparse representation, but scipy.sparse.linalg.eigs does not work
        print("dim(Ker(L1)): ", ns.shape[1])
        print("Ker(L1): ", ns)
    else:
        print("theta_0: ", initial_phase)
        print("theta_final: ", phase[:, -1])
        print(
            "theta_final: ",
            np.mod(np.around(phase[:, -1], 10), np.around(2 * np.pi, 10)),
        )

        Div = np.mod(np.around(B0.T.dot(phase), 10), np.around(2 * np.pi, 10))
        Curl = np.mod(np.around(B1.dot(phase), 10), np.around(2 * np.pi, 10))
        print("Div: ", Div[:, -1])
        print("Curl: ", Curl[:, -1])

        L1 = -B0.dot(B0.T) - B1.T.dot(B1)
        print("L1 theta: ", L1.dot(phase[:, -1]))
        print(
            "L1 theta: ",
            np.mod(np.around(L1.dot(phase[:, -1]), 10), np.around(2 * np.pi, 10)),
        )

        # w, v=eigs(L1)
        # ns_ind=np.where(w<10e-8)
        # print("dim(Ker(L1)): ", len(ns_ind)

        ns = null_space(
            L1.toarray()
        )  # not ideal, would be better in principle to stay in sparse representation, but scipy.sparse.linalg.eigs does not work
        print("dim(Ker(L1)): ", ns.shape[1])
        print("Ker(L1): ", ns)


#     plt.figure()
#     plt.imshow(Div, aspect='auto',cmap='bwr')
#     plt.title(plotname+' divergence')
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(Curl, aspect='auto',cmap='bwr')
#     plt.title(plotname+' curl')
#     plt.colorbar()


def plot_order_parameter(phases, return_op=False, plot=True):
    N = phases.shape[0]
    op = np.zeros((phases.shape[1]))
    op = np.absolute(np.exp(1j * phases).sum(0)) / N
    if plot:
        plt.figure()
        plt.plot(op)
        plt.title(op[-1])
        plt.show()
    if return_op:
        return op

def module_order_parameter(theta, community_assignement):
    
    Nc=len(np.unique(community_assignement))
    Nn=theta.shape[0]
    Nt=theta.shape[1]
    
    op=np.zeros((Nc+1,Nt))
    
    for c in range(Nc):
        ind=np.argwhere(community_assignement==c)
        op[c,:]=np.absolute(np.exp(1j*theta[ind,:]).sum(0))/len(ind)
    
    op[-1,:]=np.absolute(np.exp(1j*theta).sum(0))/Nn
    
    return op
    
def Shanahan_indices(op):
    """
    compute the two Shanahan indices
    
        l is the average across communities of the variance of the order parameter within communities ("global" metastability)
        chi is the avarage across time of the variance of the order parameter across communities at time t (Chimeraness of the system)
        op should have dimensions (number of communities+1,time), the plus one is for global order parameter on the first row
    """
    
    l = np.var(op[0:-2], axis=1).mean()
    chi = np.var(op[0:-2], axis=0).mean()
    
    return l, chi

def plot_unit_circle(phases):
    t = np.linspace(0, 2 * np.pi, 1000)
    plt.figure()
    plt.plot(np.cos(t), np.sin(t), "b", linewidth=1)
    plt.plot([0, 0], [-1, 1], "b-.", linewidth=0.5)
    plt.plot([-1, 1], [0, 0], "b-.", linewidth=0.5)
    plt.plot(np.cos(phases[:, -1]), np.sin(phases[:, -1]), "o")
    plt.show()
