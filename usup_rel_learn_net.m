%% Simple implementation of unsupervised learning of relations network
% the demo dataset contains the y = f(x) relation
%% PREPATE ENVIRONMENT
clear all; clc; close all;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL = 0;
% verbose in standard output
VERBOSE = 0;
% number of populations in the network
N_POP = 2;
% number of neurons in each population
N_NEURONS = 200;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE = 1;
% WTA circuit settling threshold
EPSILON = 1e-3;
% flag to select {epoch limit, performance limit} = {1, 0}
STOP_CRITERIA = 1;
MAX_EPOCHS = 1;
%% INIT INPUT DATA - RELATION IS EMBEDDED
MIN_VAL = 0.0; MAX_VAL = 1.0;
NUM_VALS = 10;
sensory_data.x = rand(NUM_VALS, 1)*MAX_VAL;
sensory_data.y = sensory_data.x.^2;
DATASET_LEN = length(sensory_data.x);
%% INIT NETWORK DYNAMICS
% epoch iterator in outer loop (HL, HAR)
t = 1;
% network iterator in inner loop (WTA)
tau = 1;
% constants for WTA circuit (convolution based WTA), this will provide a
% profile peaked at 0.4
DELTA = -0.005; % displacement of the convolutional kernel (neighborhood)
SIGMA = 5.0; % standard deviation in the exponential update rule (10% of N_NEURONS are close to max activation)
SL = 4.5; % scaling factor of neighborhood kernel
GAMMA = SL/(SIGMA*sqrt(2*pi)); % convolution scaling factor
% constants for Hebbian linkage
ALPHA_L = 1.0*1e-3; % Hebbian learning rate
ALPHA_D = 1.0*1e-3; % Hebbian decay factor ALPHA_D >> ALPHA_L
% constants for HAR
C = 6.0; % scaling factor in homeostatic activity regulation
TARGET_VAL_ACT = 0.4; % amplitude target for HAR
A_TARGET = TARGET_VAL_ACT*ones(N_NEURONS, 1); % HAR target activity vector
% constants for neural units in neural populations
M = 1.0; % slope in logistic function @ neuron level
S = 1.0; % shift in logistic function @ neuron level
% activity change weight (history vs. incoming knowledge)
XI = 0.05;
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
% old and new sums of weights for each population
oldWsum = zeros(N_POP, 1);
newWsum = zeros(N_POP, 1);
% quality of projection sharpness for each map
qf = zeros(N_POP, 1);
% init data index
didx = 1;
%% NETWORK SIMULATION LOOP
% % present each entry in the dataset for MAX_EPOCHS epochs to train the net
for didx = 1:DATASET_LEN
    % pick a new sample from the dataset and feed it to the input
    % population in the network (in this case X -> A -> | <- B <- Y)
    X = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
    Y = population_encoder(sensory_data.y(didx), max(sensory_data.y(:)),  N_NEURONS);
    % normalize input such that the activity in all units sums to 1.0
    X = X./sum(X);
    Y = Y./sum(Y);
    
    % clamp input to neural populations
    populations(1).a = X;
    populations(2).a = Y;
    
    % present one sample and let the network converge (HL and HAR dynamics) for
    % a given number of MAX_EPOCHS epochs or criteria is fulfilled
    while(1)
        % given the input sample wait for WTA circuit to settle
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
            populations(1).a = (1-XI)*populations(1).a + XI*delta_a1;
            populations(2).a = (1-XI)*populations(2).a + XI*delta_a2;
            
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
        end  % WTA convergence loop
        
        % update Hebbian linkage between the populations (decaying Hebbian rule)
        % compute old sum of weights
        for pidx = 1:N_POP
            oldWsum(pidx) = 1; %sum(populations(pidx).Wext(:));
        end
        
        for idx = 1:N_NEURONS
            for jdx = 1:N_NEURONS
                % compute the changes in weights
                populations(1).Wext(idx, jdx) = (1-ALPHA_D)*populations(1).Wext(idx, jdx) + ...
                    ALPHA_L*populations(1).a(idx)*populations(2).a(jdx);
                
                populations(2).Wext(idx, jdx) = (1-ALPHA_D)*populations(2).Wext(idx, jdx) + ...
                    ALPHA_L*populations(2).a(idx)*populations(1).a(jdx);
            end
        end
        
        % compute the new sums of weights
        for pidx = 1:N_POP
            newWsum(pidx) = sum(populations(pidx).Wext(:));
        end
        
        % perform inter-population Hebbian weight normalization to sum up
        % to unity in each population
        for pop_idx = 1:N_POP
            populations(pop_idx).Wext = populations(pop_idx).Wext*(oldWsum(pop_idx)/newWsum(pop_idx));
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
        
        % check the stop condition
        if(STOP_CRITERIA==1)
            if(t == MAX_EPOCHS)
                if VERBOSE==1
                    fprintf('Network after %d epochs\n', t);
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
            
            if(sum(qf)/N_POP<=EPSILON)
                if VERBOSE==1
                    fprintf('Network converged after %d epochs\n', t);
                end
                t = 1;
                % reset buffers for running average of population activities in HAR loop
                old_avg = zeros(N_POP, 1);
                break;
            end
        end
        
        % increment timestep
        t = t + 1;
        
        if VERBOSE==1
            fprintf('HL and HAR dynamics at epoch %d \n', t);
        end
    end % end main relaxation loop for HL and HAR
end % end of all samples in the training dataset

% visualize runtime data
% if(DYN_VISUAL==1)
%     visualize_runtime(sensory_data, populations, tau, t, didx);
% end

figure; set(gcf, 'color', 'white');
for pop_idx = 1:N_POP
    subplot(1, N_POP, pop_idx);
    pcolor(populations(pop_idx).Wext); caxis([0,max(populations(pop_idx).Wext(:))]); colorbar; box off;
    title(sprintf('\t Max value of W is %d | Min value of W is %d', max(populations(pop_idx).Wext(:)), min(populations(pop_idx).Wext(:))));
end