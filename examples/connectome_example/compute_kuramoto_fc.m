function FC = compute_kuramoto_fc(fun,theta0,n0)
    % Code adapted from: Pope at al, Modular origins of high-amplitude co-fluctuations in
    % fine-scale functional connectivity dynamics (2021)
    % In order to run, the original code (https://github.com/brain-networks/KSmodel_fMRIdynamics) is required

    % Returns the simulated functional connectivity matrix by computing
    % pairwise correlations of a simplicial kuramoto dynamics
    % fun: model ode right hand side
    % theta0: initial phases vector
    % n0: number of nodes

    % Code adapted from: Pope at al, Modular origins of high-amplitude co-fluctuations in
    % fine-scale functional connectivity dynamics (2021)
    % Time Parameters
    runlength = 792+20;                         % runlength (20 = length of HRF in sec, equals transient later discarded)
    transient = 20;                             % transient
    dt = 0.001;                                 % integration step
    tspan = [dt:dt:runlength];                  % solve diff equations at these time points
    tspan_transient = [dt:dt:transient];        % solve diff equations at these time points (transient)
    step = 720;                                 % TR in msec, used for downsampling Ybold
    lts = 1100;                                 % length of run in number of TRs
    
    % BOLD HRF (from BD toolbox)
    load('KSmodel_fMRIdynamics-main/BOLDHRF.mat')
    hrf = BOLDHRF(1:20000);                     % define length of hrf (20 sec)
    lhrf = length(hrf);
    
    % Max integration timestep
    odeoptions = odeset('MaxStep',dt);
        
    % Integrate ODE
    [~,theta] = ode45(fun,tspan_transient,theta0,odeoptions);
    theta = theta';

    theta0 = theta(:,end);                            % Final state of transient is IC for the following simulation
    [~,theta] = ode45(fun,tspan,theta0,odeoptions); % Integrate dynamics
    theta = theta';  
    
    % Postprocessing
    Ybold_ds = zeros(n0,lts); % Downsampled trajectories
    for n=1:n0
        Yboldtemp = conv(sin(theta(n,:)),hrf,'valid'); % BOLD signal
        Yboldtemp = lowpass(Yboldtemp,0.25,1000,'Steepness',0.95); % lowpass, 0.25 Hz
        for t=1:lts
            Ybold_ds(n,t) = mean(Yboldtemp((t-1)*step+1:t*step));
        end
    end
	
    
    Yboldglob = mean(Ybold_ds);
    Ybold_reg = zeros(lts,n0);
    for n=1:n0
        [~,~,Ybold_reg(:,n)] = regress(Ybold_ds(n,:)',[ones(1,lts)', Yboldglob']);
    end
    
    % Simulated FC = Correlation matrix
    FC = corr(Ybold_reg);
end