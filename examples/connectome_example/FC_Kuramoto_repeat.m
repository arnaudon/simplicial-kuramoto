clear
close all
clc


% Code adapted from: Pope at al, Modular origins of high-amplitude co-fluctuations in
% fine-scale functional connectivity dynamics (2021)
% In order to run, the original code (https://github.com/brain-networks/KSmodel_fMRIdynamics) is required

% Choose the model to simulate
%  model must be in {node, edge, edgeOI, nodeOM, edgeOM}
model = "edge"; 



switch model
    case "node"
        model_id = 1;
    case "edge"
        model_id = 2;
    case "edgeOI"
        model_id = 3;
    case "nodeOM"
        model_id = 4;
    case "edgeOM"
        model_id = 5;
end

%--------------------------------------------------------------------------
sc = load('KSmodel_fMRIdynamics-main/sch200_SC.mat').Kij; % Structural connectivity matrix
fc = load('KSmodel_fMRIdynamics-main/sch200_FC.mat').nFCavg; % Functional connectivity matrix
D = load('KSmodel_fMRIdynamics-main/sch200_SC.mat').D; % Delay matrix

frequency_mean = 40;     % mean natural frequency (Hz)
f_std = 0.1;   % natural frequency standard deviation
dt = 0.001;     % integration step

% Delay/Frustration Matrix
vel = 12;                                   % conduction velocity in mm/ms
D2 = D.*(1/vel);                            % delay in milliseconds
Aij = D2./((1/dt)/frequency_mean).*(2*pi);  % phase frustration
frust = tril(Aij);
frust = frust(frust~=0); % Frustration vector on the edges

%B = incidence(graph(sc)); 
B = load("boundary.mat").B; % Boundary matrix of order 1

weight = tril(sc);
weight = weight(weight~=0); % Edge weights

n0 = 200; % Number of nodes
n1 = size(B,2); % Number of edges
BW = sparse(B*diag(weight)); % Weighted boundary matrix

% Lift Matrices
V0 = sparse([eye(n0);-eye(n0)]); 
U0 = sparse([eye(n0);eye(n0)]); 
V1 = sparse([eye(n1);-eye(n1)]); 
U1 = sparse([eye(n1);eye(n1)]); 

% Operators for OI models
proj_minus = @(A) (A-abs(A))/2;
operator_frust_out = sparse(BW*proj_minus(B'*V0'));

operator_frust_out0 = sparse(proj_minus(BW*V1'));
operator_frust_in0 = sparse(V1*B');

% Partial order parameters
Rplus = @(theta) sum(cos(B'*theta))/n1; 
Rminus = @(theta) sum(cos(theta))/n0;

L0 = BW*B'; % Weighted 0-Laplacian matrix

Bfrust =BW*frust; % Projected frustration

U1frust = U1*frust; % Lifted frustration
U0Bfrust = U0*Bfrust; % Lifted projected frustration

nrep = 10; % Number of repetitions

% Get optimal coupling from previous simulations
corrs = load("correlations_final/model_correlations.mat").corrs2;
[~,idx]= max(corrs');
n_sigma = 20; 
sigma_space = linspace(100,500,n_sigma); 
sigmas_opt = sigma_space(idx); % Optimal coupling strenght for each model


corr_fc = zeros(nrep,1); % Correlations

matrices = zeros(nrep,n0,n0); % Correlation matrices


for rep = 1:nrep

    omega = (2*pi).*(frequency_mean+f_std.*randn(n0,1)); % Natural frequencies
    Bomega = BW*((2*pi).*(frequency_mean+f_std.*randn(n1,1))); % Projected natural frequencies
    theta0 = 2*pi*rand(n0,1); % Starting phases
    Btheta0 = 2*pi*BW*rand(n1,1); % Projected starting phase
    
    sigma = sigmas_opt(model_id); % Coupling strength

    node_kuramoto = @(t,theta) omega - sigma*operator_frust_out0*sin(operator_frust_in0*theta - U1frust);
    edge_kuramoto = @(t,theta) Bomega - sigma*L0*sin(theta);
    edge_kuramoto_OI = @(t,theta) Bomega - sigma*operator_frust_out*sin(V0*theta - U0Bfrust);
    node_kuramoto_OM = @(t,theta) omega - sigma*Rplus(theta)*operator_frust_out0*sin(operator_frust_in0*theta - U1frust);
    edge_kuramoto_OM = @(t,theta) Bomega - sigma*Rminus(theta)*L0*sin(theta); 
    
    disp("Rep: "+rep)
    switch model 
        case "node"
            fun = node_kuramoto;
            initial = theta0;
        case "edge"
            fun = edge_kuramoto;
            initial = Btheta0;
        case "edgeOI"
            fun = edge_kuramoto_OI;
            initial = Btheta0;
        case "nodeOM"
            fun = node_kuramoto_OM;
            initial = theta0;
        case "edgeOM"
            fun = edge_kuramoto_OM;
            initial = Btheta0;
    end
 

    % Integrate models
    FC = compute_kuramoto_fc(fun,initial,n0);
  
    corr_fc(rep) = fc_dist(FC,fc); 
    matrices(rep,:,:) = FC;
    

end

%save("correlations_final/"+model+"_correlations.mat","corr_fc")
%save("matrices_final/"+model+"_FC.mat","matrices")




