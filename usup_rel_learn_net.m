%% SIMPLE IMPLEMENTATION OF THE UNSUPERVISED LEARNING OF RELATIONS NETWORK
% the demo dataset contains the y = f(x) relation
%% PREPARE ENVIRONMENT
clear all; clc; close all;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL      = 1;
% verbose in standard output
VERBOSE         = 0;
% number of populations in the network
N_POP           = 2;
% number of neurons in each population
N_NEURONS       = 200;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE  = 1;
% WTA circuit settling threshold
EPSILON         = 1e-3;
%% INIT INPUT DATA - RELATION IS EMBEDDED IN THE INPUT DATA PAIRS
% set up the interval of interest
MIN_VAL         = -1.0; 
MAX_VAL         = 1.0;
% setup the number of random input samples to generate
NUM_VALS        = 250;
% generate NUM_VALS random samples in the given interval
sensory_data.x  = MIN_VAL + rand(NUM_VALS, 1)*(MAX_VAL - MIN_VAL);
sensory_data.y  = sensory_data.x.^2;
DATASET_LEN     = length(sensory_data.x);
%% INIT NETWORK DYNAMICS
% epoch iterator in outer loop (HL, HAR)
t       = 1;
% network iterator in inner loop (WTA)
tau     = 1;
% constants for WTA circuit (convolution based WTA), these will provide a
% profile peaked at ~ TARGET_VAL_ACT
DELTA   = -0.005;                   % displacement of the convolutional kernel (neighborhood)
SIGMA   = 5.0;                      % standard deviation in the exponential update rule 
SL      = 4.5;                      % scaling factor of neighborhood kernel
GAMMA   = SL/(SIGMA*sqrt(2*pi));    % convolution scaling factor
% constants for Hebbian linkage
ALPHA_L = 1.0*1e-2;                 % Hebbian learning rate
ALPHA_D = 1.0*1e-2;                 % Hebbian decay factor ALPHA_D >> ALPHA_L
% constants for HAR
C       = 0.005;                    % scaling factor in homeostatic activity regulation
TARGET_VAL_ACT  = 0.4;              % amplitude target for HAR
A_TARGET        = TARGET_VAL_ACT*ones(N_NEURONS, 1); % HAR target activity vector
% constants for neural units in neural populations
M       = 1; % slope in logistic function @ neuron level
S       = 10.0; % shift in logistic function @ neuron level
% activity change weight (history vs. incoming knowledge)
ETA     = 0.25;
%% CREATE NETWORK AND INITIALIZE
% create a network given the simulation constants
populations = create_init_network(N_POP, N_NEURONS, GAMMA, SIGMA, DELTA, MAX_INIT_RANGE, TARGET_VAL_ACT);
% buffers for changes in activity in WTA loop
act = zeros(N_NEURONS, N_POP)*MAX_INIT_RANGE;
old_act = zeros(N_NEURONS, N_POP)*MAX_INIT_RANGE;
% buffers for running average of population activities in HAR loop
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);
% the new rate values
delta_a1 = zeros(N_NEURONS, 1);
delta_a2 = zeros(N_NEURONS, 1);
%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:DATASET_LEN
    % pick a new sample from the dataset and feed it to the input (noiseless input)
    % population in the network (in this case X -> A -> | <- B <- Y)
    X = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
    Y = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
    % normalize input such that the activity in all units sums to 1.0
    X = X./sum(X);
    Y = Y./sum(Y);
    % clamp input to neural populations
    populations(1).a = X;
    populations(2).a = Y;
    % given the input sample wait for WTA circuit to settle and then
    % perform a learning step of Hebbian learning and HAR
    while(1)
        % for each neuron in first population
        for idx = 1:N_NEURONS
            % save the between populations contribution
            ext_contrib = 0.0;
            % save the within population contribution
            int_contrib = 0.0;
            for jdx = 1:N_NEURONS
                % compute between populations contribution
                ext_contrib = ext_contrib + populations(1).Wext(idx, jdx)*populations(2).a(jdx);
                % compute the contribution within population
                if(idx~=jdx)
                    int_contrib = int_contrib + populations(1).Wint(idx, jdx)*populations(1).a(jdx);
                end
            end
            % update the delta
            delta_a1(idx) = compute_s(populations(1).h(idx) + ext_contrib + int_contrib, M, S);
        end                    
        % for each neuron in the second population
        for idx = 1:N_NEURONS
            % save the between populations contribution
            ext_contrib = 0.0;
            % save the within population contribution
            int_contrib = 0.0;
            for jdx = 1:N_NEURONS
                % compute between populations contribution
                ext_contrib = ext_contrib + populations(2).Wext(idx, jdx)*populations(1).a(jdx);
                % compute the contribution within population
                if(idx~=jdx)
                    int_contrib = int_contrib + populations(2).Wint(idx, jdx)*populations(2).a(jdx);
                end
            end
            % update the delta
            delta_a2(idx) = compute_s(populations(2).h(idx) + ext_contrib + int_contrib, M, S);
        end
        % update the activities of each population
        populations(1).a = (1-ETA)*populations(1).a + ETA*delta_a1;
        populations(2).a = (1-ETA)*populations(2).a + ETA*delta_a2;
        % current activation values holder
        for pop_idx = 1:N_POP
            act(:, pop_idx) = populations(pop_idx).a;
        end
        % check if activity has settled in the WTA loop
        q = (sum(sum(abs(act - old_act)))/(N_POP*N_NEURONS));
        if(q <= EPSILON)
            if VERBOSE==1
                fprintf('WTA converged after %d iterations\n', tau);
            end
            tau = 1;
            break;  
        end
        % update history of activities
        old_act = act;
        % increment time step in WTA loop
        tau = tau + 1;
        % visualize runtime data
        if(DYN_VISUAL==1)
            visualize_runtime(sensory_data, populations, tau, t, didx);
        end
    end  % WTA convergence loop
    % update Hebbian linkage between the populations (decaying Hebbian rule)
    for idx = 1:N_NEURONS
        for jdx = 1:N_NEURONS
            % compute the changes in weights
            populations(1).Wext(idx, jdx) = (1-ALPHA_D)*populations(1).Wext(idx, jdx) + ...
                ALPHA_L*populations(1).a(idx)*populations(2).a(jdx);
            
            populations(2).Wext(idx, jdx) = (1-ALPHA_D)*populations(2).Wext(idx, jdx) + ...
                ALPHA_L*populations(2).a(idx)*populations(1).a(jdx);
        end
    end
    % compute the inverse time for exponential averaging of HAR activity
    omegat = 0.002 + 0.998/(t+2);
    % for each population in the network
    for pop_idx = 1:N_POP
        for idx = 1:N_NEURONS
            % update Homeostatic Activity Regulation terms
            % compute exponential average of each population at current step
            cur_avg(pop_idx, idx) = (1-omegat)*old_avg(pop_idx, idx) + omegat*populations(pop_idx).a(idx);
            % update homeostatic activity terms given current and target act.
            populations(pop_idx).h(idx) = populations(pop_idx).h(idx) + C*(TARGET_VAL_ACT - cur_avg(pop_idx, idx));
        end
    end
    % update averging history
    old_avg = cur_avg;
    % increment timestep for HL and HAR loop
    t = t + 1;
    % print epoch counter
    if VERBOSE==1
        fprintf('HL and HAR dynamics at iteration %d \n', t);
    end
end % end of all samples in the training dataset
% visualize post-runtime data
visualize_runtime(sensory_data, populations, 1, t, DATASET_LEN);
