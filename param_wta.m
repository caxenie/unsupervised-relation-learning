% simple script to parametrize the WTA circuit in the sharp learning net

% in each population the neurons are laterally interconnected to implement
% a soft Winner-Take-All circuit

% test the parametrization of a WTA 
close all; clear all; clc; 

% number of neurons in the population 
N_NEURONS = 200;

% standard deviation 
SIGMA = 5.0;
% scaling factor
SL = 5.0; % original value 4.5
% convolution scaling factor 
GAMMA = SL/(SIGMA*sqrt(2*pi));
% decay term 
DELTA = -0.001; % original value -0.005

% the weight matrix for lateral interconnectivity 
W = rand(N_NEURONS, N_NEURONS);

% build up the connectivity matrix 
for idx = 1:N_NEURONS
    for jdx = 1:N_NEURONS
        dij = min([abs(idx-jdx), N_NEURONS - abs(idx-jdx)]);
        W(idx, jdx) = GAMMA*exp(-0.5*(dij/SIGMA)^2) - DELTA;       
    end
end

% visualize the lateral connectivity weight matrix
figure; set(gcf, 'color', 'w');
subplot(1,2,1);
pcolor(W); box off; grid off; colorbar;
xlabel('neuron index j'); ylabel('neuron index i');
subplot(1,2,2);
surf(W(1:N_NEURONS, 1:N_NEURONS)); box off; grid off;
xlabel('neuron index j'); ylabel('neuron index i');
zlabel('synaptic strength');