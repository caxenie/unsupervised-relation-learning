% compute the distance between the locations of 2 neurons in a population
% used for the convolutional based WTA implementation
function y = compute_d(N, s)
    y = zeros(N, N);
    for idx = 1:N
        for jdx = 1:N
            y(idx, jdx) = exp(-0.5*(min([abs(idx-jdx), N - abs(idx-jdx)])/s)^2);
        end
    end
end