clear
close all
clc


% Find optimal coupling strength for each model

% Code adapted from: Pope at al, Modular origins of high-amplitude co-fluctuations in
% fine-scale functional connectivity dynamics (2021)
% In order to run, the original code (https://github.com/brain-networks/KSmodel_fMRIdynamics) is required

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

n_sigma = 20; 
sigma_space = linspace(100,500,n_sigma); % Coupling strengths to scan
n_models = 5;

corr_fc = zeros(n_models,n_sigma); % Correlations

for s = 1:n_sigma
    omega = (2*pi).*(frequency_mean+f_std.*randn(n0,1)); % Natural frequencies
    Bomega = BW*((2*pi).*(frequency_mean+f_std.*randn(n1,1))); % Projected natural frequencies
    theta0 = 2*pi*rand(n0,1); % Starting phases
    Btheta0 = 2*pi*BW*rand(n1,1); % Projected starting phase
    
    sigma = sigma_space(s); % Coupling strength
    
    node_kuramoto = @(t,theta) omega - sigma*operator_frust_out0*sin(operator_frust_in0*theta - U1frust);
    edge_kuramoto = @(t,theta) Bomega - sigma*L0*sin(theta);
    edge_kuramoto_OI = @(t,theta) Bomega - sigma*operator_frust_out*sin(V0*theta - U0Bfrust);
    node_kuramoto_OM = @(t,theta) omega - sigma*Rplus(theta)*operator_frust_out0*sin(operator_frust_in0*theta - U1frust);
    edge_kuramoto_OM = @(t,theta) Bomega - sigma*Rminus(theta)*L0*sin(theta); 
    

    % Integrate models
    node_FC = compute_kuramoto_fc(node_kuramoto,theta0,n0);
    corr_fc(1,s) = fc_dist(node_FC,fc); 
    disp("s: "+s+"/"+n_sigma+" node")

    edge_FC = compute_kuramoto_fc(edge_kuramoto,Btheta0,n0);
    corr_fc(2,s) = fc_dist(edge_FC,fc); 
    disp("s: "+s+"/"+n_sigma+" edge")

    edgeOI_FC = compute_kuramoto_fc(edge_kuramoto_OI,Btheta0,n0);
    corr_fc(3,s) = fc_dist(edgeOI_FC,fc); 
    disp("s: "+s+"/"+n_sigma+" edgeOI")
    
    nodeOM_FC = compute_kuramoto_fc(node_kuramoto_OM,theta0,n0);
    corr_fc(4,s) = fc_dist(nodeOM_FC,fc); 
    disp("s: "+s+"/"+n_sigma+" nodeOM")

    edgeOM_FC = compute_kuramoto_fc(edge_kuramoto_OM,Btheta0,n0);
    corr_fc(5,s) = fc_dist(edgeOM_FC,fc); 
    disp("s: "+s+"/"+n_sigma+" edgeOM")

end

save("correlations_final/model_correlations.mat","corr_fc")




