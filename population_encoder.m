%% Poisson Noisy Pattern of activity for Input Neurons
%   This function generates a Poisson Noisy pattern of activity
%   for each neuron in population as Initial tuning curve 
%   Variables and Argument Description:
%   x:      input analog value subjected to be encoded into ...
%           ... activity pattern, dimension in (Radians)
%   sig:    Defines the spread in radians (Rad)
%   v:      spontaneous level of activity for each neuron (Hz)
%   N:      number of neurons in the population
%   C:      A factor in sec, shows the time duration in which the ...
%           activity of neurons has been considered
function R = population_encoder(x, N) % , sig, v, C)
sig = 0.4;
v = 1;
C = 1;
K = 20; % in Hz

R = zeros(N, 1);     % Pattern of activity, or output tuning curve
for j = 1:1:N
    temp = cos( x - (2*pi*j/N) ) - 1 ;
    % fj is the lamda value for poisson Neuron j
    fj = C * (K*exp(temp / sig^2) + v); 
    %R(j) = poissrnd(fj);
    R(j) = fj;
end

    
    
    
    