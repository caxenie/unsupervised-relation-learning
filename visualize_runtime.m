% function to visualize network data at a given iteration in runtime
function visualize_runtime(sensory_data, populations, tau, t, d)
set(gcf, 'color', 'w');
% activities for each population (both overall activity and homeostasis)
subplot(3, 3, 1);
acth3 = plot(population_encoder(sensory_data.x(d), max(sensory_data.x(:)),  populations(1).lsize), '--r', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('pc coded input in layer 1');
subplot(3, 3, 2);
acth4 = plot(population_encoder(sensory_data.y(d), max(sensory_data.y(:)),  populations(1).lsize), '--b', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('pc coded input in layer 2'); title(sprintf('Network dynamics: tau = %d (WTA) | t = %d (HL, HAR)', tau, t));
subplot(3, 3, 3);
acth5 = plot(sensory_data.x(d), sensory_data.y(d), '*k'); hold on; plot(sensory_data.x, sensory_data.y, '-g'); box off;
xlabel('X'); ylabel('Y');

subplot(3, 3, 4);
acth6 = plot(populations(1).a, '-r', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 1');
subplot(3, 3, 5);
acth7 = plot(populations(2).a,  '-b', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 2');
subplot(3, 3, 6);
acth8 = pcolor(populations(1).Wint); 
xlabel('neuron index'); ylabel('neurons index'); 

% hebbian links between populations
subplot(3, 3, 7);
acth9 = pcolor(populations(1).Wext);
box off; grid off;
xlabel('layer 1 - neuron index'); ylabel('layer 2 - neuron index');
subplot(3, 3, 8);
acth10= pcolor(populations(2).Wext);
box off; grid off;
xlabel('layer 2 - neuron index'); ylabel('layer 1 - neuron index');

% refresh visualization
set(acth3, 'YData', population_encoder(sensory_data.x(d), max(sensory_data.x(:)),  populations(1).lsize));
set(acth4, 'YData', population_encoder(sensory_data.y(d), max(sensory_data.y(:)),  populations(1).lsize));
set(acth5, 'XData', sensory_data.x(d));
set(acth5, 'YData', sensory_data.y(d));
set(acth6, 'YDataSource', 'populations(1).a'); 
set(acth7, 'YDataSource','populations(2).a'); 
set(acth8, 'CData', populations(1).Wint);
set(acth9, 'CData', populations(1).Wext);
set(acth10,'CData', populations(2).Wext);

drawnow;
end