"""Numerical integrators."""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def node_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None):
    """Node simplicial kuramoto, or classical Kuramoto."""
    B0 = simplicial_complex.B0
    W0 = simplicial_complex.W0
    W1 = simplicial_complex.W1

    if omega_0 is None:
        omega_0 = np.zeros(simplicial_complex.n_nodes)

    W1_inv = W1.copy()
    W1_inv.data = 1./ W1_inv.data
    return omega_0 - W0.dot(B0.T).dot(np.sin(W1_inv.dot(B0).dot(phase)))


def integrate_node_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, omega_0=None
):
    """Integrate the node Kuramoto model."""
    rhs = partial(
        node_simplicial_kuramoto, simplicial_complex=simplicial_complex, omega_0=omega_0
    )
    t_eval = np.linspace(0, t_max, n_t)
    opt = solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )    
    
    opt.y[opt.y<0] += 2*np.pi
    opt.y[opt.y>2*np.pi] -= 2*np.pi
    
    return opt

def edge_simplicial_kuramoto(time, phase, simplicial_complex=None, omega_0=None):
    """Edge simplicial kuramoto"""
    B0 = simplicial_complex.B0
    W0 = simplicial_complex.W0
    B1 = simplicial_complex.B1
    W1 = simplicial_complex.W1
    W2 = simplicial_complex.W2

    if omega_0 is None:
        omega_0 = np.zeros(simplicial_complex.n_edges)

    W1_inv = W1.copy()
    W1_inv.data = 1./ W1_inv.data
    rhs = B0.dot(W0).dot(np.sin(B0.T.dot(W1_inv).dot(phase)))

    if W2 is not None:
        W2_inv = W2.copy()
        W2_inv.data = 1./ W2_inv.data
        rhs += W1.dot(B1.T).dot(np.sin(W2_inv.dot(B1).dot(phase)))
    return omega_0 - rhs


def integrate_edge_kuramoto(
    simplicial_complex, initial_phase, t_max, n_t, omega_0=None
):
    """Integrate the edge Kuramoto model."""

    rhs = partial(
        edge_simplicial_kuramoto, simplicial_complex=simplicial_complex, omega_0=omega_0
    )
    t_eval = np.linspace(0, t_max, n_t)
    opt =  solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    opt.y[opt.y<0] += 2*np.pi
    opt.y[opt.y>2*np.pi] -= 2*np.pi
    
    return opt

def node_simplicial_kuramoto_frustrated(time, phase, simplicial_complex=None, alpha_0=None, alpha_1=None):
    """Node simplicial kuramoto, or classical Kuramoto."""
    B0 = simplicial_complex.B0
    LB0 = simplicial_complex.lifted_B0
    LB0p = simplicial_complex.lifted_B0_p
    W0 = simplicial_complex.W0
    W1 = simplicial_complex.W1

    if alpha_0 is None:
        alpha_0=np.zeros(simplicial_complex.n_nodes)
        
    if alpha_1 is None:
        alpha_1=np.zeros(2*simplicial_complex.n_edges)
    else:
        alpha_1_v=simplicial_complex.V.dot(alpha_1)

    rhs=-alpha_0-LB0p.T.dot(np.sin(LB0.dot(phase)-alpha_1_v))

    return rhs

def integrate_node_kuramoto_frustrated(
    simplicial_complex, initial_phase, t_max, n_t, alpha_0=None, alpha_1=None
):
    """Integrate the node Kuramoto model."""
    rhs = partial(
        node_simplicial_kuramoto_frustrated, simplicial_complex=simplicial_complex, alpha_0=alpha_0, alpha_1=alpha_1
    )
    t_eval = np.linspace(0, t_max, n_t)
    opt = solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )    
    
    opt.y[opt.y<0] += 2*np.pi
    opt.y[opt.y>2*np.pi] -= 2*np.pi
    
    return opt

def edge_simplicial_kuramoto_frustrated(time, phase, simplicial_complex=None, alpha_1=None, alpha_2=None):
    B0 = simplicial_complex.B0
    W0 = simplicial_complex.W0
    B1 = simplicial_complex.B1
    W1 = simplicial_complex.W1
    W2 = simplicial_complex.W2
    LB0 = simplicial_complex.lifted_B0
    LB1 = simplicial_complex.lifted_B1
    LB0p = simplicial_complex.lifted_B0_p
    LB1p = simplicial_complex.lifted_B1_p
    
    Nn=B0.shape[1]
    Ne=B0.shape[0]
    Nf=B1.shape[0]
    
    if alpha_1 is None:
        alpha_1=np.zeros(2*simplicial_complex.n_edges)
    else:
        alpha_1_v=np.ones(2*Ne)*alpha_1
        
    if alpha_2 is None:
        alpha_2=np.zeros(2*simplicial_complex.n_edges)
    else:
        alpha_2_v=np.ones(Nf)*alpha_2
        
    rhs=-LB0.dot(np.sin(LB0p.T.dot(phase)))-alpha_1_v

    if W2 is not None:
        rhs += -LB1p.T.dot(np.sin(LB1.dot(phase)+alpha_2_v))
    
    return rhs

def integrate_edge_kuramoto_frustrated(
    simplicial_complex, initial_phase, t_max, n_t, alpha_1=None, alpha_2=None
): 
    """Integrate the frustrated edge Kuramoto model."""

    rhs = partial(
        edge_simplicial_kuramoto_frustrated, simplicial_complex=simplicial_complex, alpha_1=alpha_1, alpha_2=alpha_2
    )
    t_eval = np.linspace(0, t_max, n_t)
    opt =  solve_ivp(
        rhs,
        [0, t_max],
        initial_phase,
        t_eval=t_eval,
        method="Radau",
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    opt.y[opt.y<0] += 2*np.pi
    opt.y[opt.y>2*np.pi] -= 2*np.pi
    
    return opt