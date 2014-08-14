%% Simple implementation of unsupervised learning of relations network
% the demo dataset contains the y = x^3 relation for x in [1, 10]

%% PREPATE ENVIRONMENT
clear all; clc; close all;

%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL = 1;
% verbose in standard output
VERBOSE = 0;
% number of populations in the network
N_POP = 2;
% number of neurons in each population
N_NEURONS = 50;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE = 1;
% WTA circuit settling threshold
EPSILON = 1e-12;
% number of epochs to train the nework
% flag to select {epoch limit, performance limit} = {1, 0}
STOP_CRITERIA = 1;
MAX_EPOCHS = 50;

%% INIT INPUT DATA - RELATION IS EMBEDDED
MIN_VAL = -1.0; MAX_VAL = 1.0;
sensory_data.x = linspace(MIN_VAL, MAX_VAL, N_NEURONS);
sensory_data.y = sensory_data.x.^3;
DATASET_LEN = N_NEURONS;

%% INIT NETWORK DYNAMICS
% epoch iterator in outer loop (HL, HAR)
t = 1;
% network iterator in inner loop (WTA)
tau = 1;
% constants for WTA circuit (convolution based WTA)
DELTA = -0.005; % displacement of the convolutional kernel (neighborhood)
SIGMA = 5.0; % standard deviation in the exponential update rule
SL = 4.5; % scaling factor of neighborhood kernel
GAMMA = SL/(SIGMA*sqrt(2*pi)); % convolution scaling factor
% constants for Hebbian linkage
ALPHA_L = 0.5*1e-3; % Hebbian learning rate
ALPHA_D = 0.5*1e-3; % Hebbian decay factor ALPHA_D > ALPHA_L
% constants for HAR
C = 6.0; % scaling factor in homeostatic activity regulation
TARGET_VAL_ACT = 0.4; % amplitude target for HAR
A_TARGET = TARGET_VAL_ACT*ones(N_NEURONS, 1); % HAR target activity vector
% constants for neural units in neural populations
M = 1.0; % slope in logistic function @ neuron level
S = 1.55; % shift in logistic function @ neuron level

%% CREATE NETWORK AND INITIALIZE
% create a network given the simulation constants
populations = create_init_network(N_POP, N_NEURONS, GAMMA, SIGMA, DELTA, MAX_INIT_RANGE, TARGET_VAL_ACT);
% buffers for changes in activity in WTA loop
act = rand(N_NEURONS, N_POP)*MAX_INIT_RANGE;
old_act = rand(N_NEURONS, N_POP)*MAX_INIT_RANGE;
% buffers for running average of population activities in HAR loop
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);
% quality of projection sharpness for each map
qf = zeros(N_POP, 1);

%% NETWORK SIMULATION LOOP
% present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:DATASET_LEN
    
    % pick a new sample from the dataset and feed it to the input
    % population in the network (in this case in->A-> | <- B<- in)
    old_act(:, 1) = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
    old_act(:, 2) = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
    
    % present one sample and let the network converge (HL and HAR dynamics)
    while(1)
        % given the input sample wait for WTA circuit to settle
        while(1)
            % neural units activity update for each population
            populations(1).a = compute_s(populations(1).h + ...
                populations(1).Wint*old_act(:, 1) + ...
                populations(1).Wext*old_act(:, 2), M, S);
            
            populations(2).a = compute_s(populations(2).h + ...
                populations(2).Wint*old_act(:, 2) + ...
                populations(2).Wext*old_act(:, 1), M, S);
            
            % current activation values holder
            for pop_idx = 1:N_POP
                act(:, pop_idx) = populations(pop_idx).a;
            end
            
            % check if activity has settled in the WTA loop
            if((sum(sum(abs(act - old_act)))/(N_POP*N_NEURONS))<EPSILON)
                if VERBOSE==1
                    fprintf('Network converged after %d iterations\n', tau);
                end
                tau = 1;
                break;
            end
            
            % update history of activities
            old_act = act;
            % increment time step in WTA loop
            tau = tau + 1;
            
        end  % WTA convergence loop
        
        % update Hebbian linkage between the populations (decaying Hebbian rule)
        populations(1).Wext = (1-ALPHA_D)*populations(1).Wext + ...
            ALPHA_L*populations(2).a*populations(1).a';
        
        populations(2).Wext = (1-ALPHA_D)*populations(2).Wext + ...
            ALPHA_L*populations(1).a*populations(2).a';
        
        % compute the inverse time for exponential averaging of HAR activity
        omegat = 0.002 + 0.998/(t+2);
        
        % for each population in the network
        for pop_idx = 1:N_POP
            % perform inter-population Hebbian weight normalization
            populations(pop_idx).Wext = populations(pop_idx).Wext./max(populations(pop_idx).Wext(:));
            
            % update Homeostatic Activity Regulation terms
            % compute exponential average of each population at current step
            cur_avg(pop_idx, :) = (1-omegat)*old_avg(pop_idx, :) + omegat*populations(pop_idx).a';
            % update homeostatic activity terms given current and target act.
            populations(pop_idx).h = -C*(cur_avg(pop_idx, :)' - A_TARGET);
        end
        
        % update averging history
        old_avg = cur_avg;
        
        % check criteria to stop learning on the current sample
        if(STOP_CRITERIA==1)
            % max epochs reached
            if(t == MAX_EPOCHS)
                if VERBOSE==1
                    fprintf('Network converged after %d epochs\n', t);
                end
                t = 1;
                break;
            end
        else 
            % compute quality function for each population
            diff_vector = zeros(2, 2);
            for pidx = 1:N_POP
                diff_vector(pidx, :) = [abs(compute_c(populations(pidx)) - didx), N_NEURONS - abs(compute_c(populations(pidx)) - didx)];
                qf(pidx) = 2/N_NEURONS*(min(diff_vector(pidx, :)));
            end
            
            if(1/N_POP*sum(qf)<=EPSILON)
                if VERBOSE==1
                    fprintf('Network converged after %d epochs\n', t);
                end
                t = 1;
                break;
            end
        end % end if selection of stop criteria
        
        % increment timestep
        t = t + 1;
        
        % if runtime visualization is enable - slows down the simulation
        if(DYN_VISUAL==1)
            visualize_runtime(sensory_data, populations, tau, t, didx);
        end
        
    end % end main relaxation loop for HL and HAR
end % end of all samples in the training dataset