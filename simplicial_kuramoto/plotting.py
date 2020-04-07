"""Plotting functions."""
import matplotlib.pyplot as plt


def plot_node_kuramoto(node_results):
    """Basic plot for node kuramoto."""
    plt.figure()
    plt.imshow(node_results.y, aspect="auto")
    plt.xlabel("time")
    plt.ylabel("mode id")

def plot_flow(initial_phase,simpicial_somplex,result,plotname=None):
    """ Some outputs (curl, div, etc ...) are only useful for understanding what is happening """
    times = result.t
    phase = result.y
    
    B0=simplicial_complex.node_incidence_matrix
    B1=simplicial_complex.edge_incidence_matrix
    
    plt.figure()
    plt.imshow(np.mod(np.around(phase,10),np.around(2*np.pi,10)), aspect='auto',cmap='bwr')
    plt.title(plotname+' phases')
    plt.colorbar()
    
    op=order_parameter(phase, 4, 1) # that is in the utils.py
    plt.figure()
    plt.title(plotname+' order parameter')
    plt.plot(op[0,:])

    print('\theta_0: ', initial_phase)
    print('\theta_final: ',phase[:,-1])
    print('\theta_final: ',np.mod(np.around(phase[:,-1],10),np.around(2*np.pi,10)))
    
    Div=np.mod(np.around(B0.T.dot(phase),10),np.around(2*np.pi,10))
    Curl=np.mod(np.around(B1.dot(phase),10),np.around(2*np.pi,10))
    print('Div: ', Div[:,-1])
    print('Curl: ', Curl[:,-1])
    
    L1=-B0.dot(B0.T)-B1.T.dot(B1)
    print('L1\theta: ', L1.dot(phase[:,-1]))
    print('L1\theta: ', np.mod(np.around(L1.dot(phase[:,-1]),10),np.around(2*np.pi,10)))
    print('dim(Ker(L1)): ', null_space(L1).shape[1])
    print('Ker(L1): ', null_space(L1))
    
#     plt.figure()
#     plt.imshow(Div, aspect='auto',cmap='bwr')
#     plt.title(plotname+' divergence')
#     plt.colorbar()
#     plt.figure()
#     plt.imshow(Curl, aspect='auto',cmap='bwr')
#     plt.title(plotname+' curl')
#     plt.colorbar()
