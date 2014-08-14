% extract the position of the center of the relaxed activity pattern in a
% population using population vector decoding
function x_star = compute_c(population)
    sum_pop_vect = 0.0;
    for idx = 1:population.lsize
        sum_pop_vect = sum_pop_vect + population.a(idx)*exp(2*1i*idx/population.lsize);
    end
    x_star = phase(sum_pop_vect)*(population.lsize/2*pi);
end