% compute the value of the logistic function for single neuron dynamics
% given the slope, m, and the shift, s
function y = compute_s(x, m, s)
    y = zeros(length(x), 1);
    for idx = 1:length(x)
        y(idx) = 1/(1 + exp(-m*(x(idx) - s)));
    end
end