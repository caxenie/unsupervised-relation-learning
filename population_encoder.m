% function to generate the population encoded variable as input for the net
% here we also need to encode variables which are in both +/- ranges
% we need to take into accound the encoding for the tuning curves
% distribution
function R = population_encoder(x, range, N)
sig = 0.1; % standard deviation 
K = 1; % max firing rate (Hz) (ignore - not modeling nurophysiology here :)
% pattern of activity, or output tuning curve between [-range, range]
R = zeros(N, 1);
% calculate output 
for j = 1:N % for each neuron in the population
    R(j) = K*exp( -(x - (-range+(j)*(range/((N)/2))))^2 / (2*sig^2));
end

    
    
    
    