% crate the sharp learning network composed of N_POP populations of
% N_NEURONS neurons and init each struct weight and activity matrices
function populations = create_init_network(N_POP, N_NEURONS, GAMMA, SIGMA, DELTA, MAX_INIT_RANGE, TARGET_VAL_ACT)
    wta_profile = GAMMA*compute_d(N_NEURONS, SIGMA) - DELTA;
    for pop_idx = 1:N_POP
        populations(pop_idx) = struct(  'idx', pop_idx,...
            'lsize', N_NEURONS, ...
            'Wint', (wta_profile)./max(wta_profile(:)), ...
            'Wext', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
            'a', zeros(N_NEURONS, 1)*TARGET_VAL_ACT,...
            'h', rand(N_NEURONS, 1)*TARGET_VAL_ACT);
    end
end