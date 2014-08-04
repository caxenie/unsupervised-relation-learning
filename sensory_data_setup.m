function data = sensory_data_setup(embedded, external)
format long g;

% load sensory data
robot_data   = load(embedded);
vision_data  = load(external);

% some global consts of the system
imu_scaling_factor = 1000;      % samples
magneto_scaling_factor = 1000;  % samples
omnirob_buffer_size = 8;        % samples
sample_freq = 25;               % Hz
sample_time = 1/sample_freq;    % s
robot_wheel_radius = 0.027;     % m
radius_t = 0.032;               % m
robot_base = 0.082;             % m
wrap_up_compass = 180;          % deg

% for sensory data analysis use heading from cam
ref_sys_rotation = deg2rad(vision_data(:, 4));

%% Datasets alignment

% tolerance in timestamp comparison
tolerance = 1e-18;

% get the start and stop timestamp in both datasets
start_ts_vision = min(vision_data(:,1));
start_ts_robot   = min(robot_data(:,1));
stop_ts_vision  = max(vision_data(:,1));
stop_ts_robot    = max(robot_data(:,1));

% compute the global start and stop time for the data
global_start = max(start_ts_robot, start_ts_vision);
global_stop  = min(stop_ts_robot, stop_ts_vision);

% select and remove out-of-range entries for robot sensory dataset
pruned_entries_robot = [];
pruned_idx_robot = 1;
for i=1:length(robot_data)
    % start check
    if (robot_data(i,1) - global_start < tolerance)
        pruned_entries_robot(pruned_idx_robot) = i;
        pruned_idx_robot = pruned_idx_robot + 1;
    end
    % end check
    if (robot_data(i,1) - global_stop > tolerance)
        pruned_entries_robot(pruned_idx_robot) = i;
        pruned_idx_robot = pruned_idx_robot + 1;
    end
end
if(isempty(pruned_entries_robot)==0)
    robot_data = removerows(robot_data, 'ind', pruned_entries_robot);
end

% select and remove out-of-range entries for camera sensory dataset
pruned_entries_vision = [];
pruned_idx_vision = 1;
for i=1:length(vision_data)
    % start check
    if (vision_data(i,1) - global_start < tolerance)
        pruned_entries_vision(pruned_idx_vision) = i;
        pruned_idx_vision = pruned_idx_vision + 1;
    end
    % end check
    if (vision_data(i,1) - global_stop >  tolerance)
        pruned_entries_vision(pruned_idx_vision) = i;
        pruned_idx_vision = pruned_idx_vision + 1;
    end
end
if(isempty(pruned_entries_vision)==0)
    vision_data = removerows(vision_data, 'ind', pruned_entries_vision);
end

% check the sizes after pruning the out-of-bound entries
if (length(robot_data) == length(vision_data))
    % go on with analysis
else
    pruning_cand_robot = [];
    pruning_cand_robot_idx = 1;
    pruning_cand_vision = [];
    pruning_cand_vision_idx = 1;
    % check which dataset is longer and check the timestamp diff
    if (length(robot_data) > length(vision_data))
        % robot dataset is bigger so shrink it
        % compare the beggining and the end of the vision dataset until we
        % reach same size
        for i=1:length(robot_data)
            if(robot_data(i,1) - vision_data(1,1) <  tolerance)
                pruning_cand_robot(pruning_cand_robot_idx) = i;
                pruning_cand_robot_idx = pruning_cand_robot_idx + 1;
            end
            if(robot_data(i,1) - vision_data(length(vision_data),1) > tolerance)
                pruning_cand_robot(pruning_cand_robot_idx) = i;
                pruning_cand_robot_idx = pruning_cand_robot_idx + 1;
            end
        end
    else
        % vision dataset is bigger so shrink it
        % compare the beggining and the end of the vision dataset until we
        % reach same size
        for i=1:length(vision_data)
            if(vision_data(i,1) - robot_data(1,1) <  tolerance)
                pruning_cand_vision(pruning_cand_vision_idx) = i;
                pruning_cand_vision_idx = pruning_cand_vision_idx + 1;
            end
            if(vision_data(i,1) - robot_data(length(robot_data),1) > tolerance)
                pruning_cand_vision(pruning_cand_vision_idx) = i;
                pruning_cand_vision_idx = pruning_cand_vision_idx + 1;
            end
        end
    end
    robot_data = removerows(robot_data, 'ind', pruning_cand_robot);
    vision_data = removerows(vision_data, 'ind', pruning_cand_vision);
    
    % final cuts to fit the size
    diff_rt = length(robot_data) - length(vision_data);
    diff_tr = length(vision_data) - length(robot_data);
    if(sign(diff_rt)==1)
        for i=1:diff_rt
            robot_data = removerows(robot_data, 'ind', length(robot_data)-i);
        end
    end
    if(sign(diff_tr)==1)
        for i=1:diff_tr
            vision_data = removerows(vision_data, 'ind', length(vision_data)-i);
        end
    end
    
end

%% Sensory data analysis for heading estimation

time_units = [1:length(robot_data)]./sample_freq;

%% Vision data analysis

% clean vision noise at startup (spurios jumps of the tracking software)
movement_limit_vision = 350;
for i=2:length(vision_data)
    delta = vision_data(i,4) - vision_data(i-1, 4);
    if(delta >= movement_limit_vision)
        vision_data(i,4) = vision_data(i-1,4);
    end
end

% compute vision offset
vision_offset = vision_data(2,4);
% if internal preprocessing is disabled

% plot robot trajectory from vision
% figure;
% set(gcf, 'color', 'w'); box off;
% subplot(441);
% plot(vision_data(:,2), vision_data(:,3), '-g', 'LineWidth', 4);
% title('Robot trajectory from vision'); grid on; set(gca, 'Box', 'off');
% axis_handle_proc(1) = subplot(442);
%remove vision offset
head_vision = vision_data(:,4) - vision_offset;
% plot(time_units, head_vision,'.g');
% xlabel('Time(s)');
% title('Absolute vision angle');grid on; set(gca, 'Box', 'off');


%% Odometry data analysis

% compute change in heading angle from odometry
dhead_odometry = zeros(1, length(robot_data));
for i=1:length(robot_data)
    dhead_odometry(i) = (((pi/30) * (robot_data(i,2)/omnirob_buffer_size + ...
        robot_data(i,3)/omnirob_buffer_size + ...
        robot_data(i,4)/omnirob_buffer_size)*robot_wheel_radius)/...
        (3*robot_base))*(180/pi);
end

% compute the integrated angle from odometry if preprocessing not embedded
head_odometry = zeros(1, length(dhead_odometry));

dhead_odometry_ant = dhead_odometry(1);
for i=2:length(robot_data)
    head_odometry(i) = head_odometry(i-1) + ...
        (sample_time*(dhead_odometry_ant + dhead_odometry(i))*.5);
    dhead_odometry_ant = dhead_odometry(i);
end
% plot data from odometry
% axis_handle_raw(2) = subplot(445); plot(time_units, dhead_odometry,'.b');
% title('Raw odometry change of angle'); grid on; set(gca, 'Box', 'off');
% axis_handle_proc(2) = subplot(446); plot(time_units, head_odometry, '.b');
% title('Absolute odometry angle'); grid on; set(gca, 'Box', 'off');


%% Gyroscope data analysis

% scale the data and convert from rad/s to deg/s
dhead_gyro = (robot_data(:,7)*(180/pi))/imu_scaling_factor ;
% make consistent with robot reference frame
dhead_gyro = -dhead_gyro;
% init history of integrated angle value
head_gyro = zeros(1, length(robot_data));

% if preprocessing is not embedded in the net

% init integration
head_ant = dhead_gyro(1);
% compute the absolute angle from the gyro
for i=2:length(robot_data)
    head_gyro(i) = head_gyro(i-1) + ...
        (sample_time*(head_ant + dhead_gyro(i))*.5);
    head_ant = dhead_gyro(i);
end
% plot data from gyroscope
% axis_handle_raw(3) = subplot(449); plot(time_units, dhead_gyro, '.r');
% title('Raw gyro change of angle'); grid on; set(gca, 'Box', 'off');
% axis_handle_proc(3) = subplot(4,4,10); plot(time_units, head_gyro, '.r');
% title('Absolute gyro angle'); grid on; set(gca, 'Box', 'off');


%% Magneto analysis

% adjust the sign of the magneto to be compliant with the other sensors
robot_data(:,14) = -(robot_data(:,14));
magneto_raw    = robot_data(:,14)/magneto_scaling_factor;
magneto_offset = magneto_raw(1);
magneto_aligned = magneto_raw(:) - magneto_offset;
% handle jumps and wrap-ups
for i=2:length(magneto_aligned)
    delta = magneto_aligned(i-1) - magneto_aligned(i);
    if(delta > wrap_up_compass)
        magneto_aligned(i) = magneto_aligned(i) + 2*wrap_up_compass;
    end
    
end
head_magneto = magneto_aligned;
% if preprocessing is not embedded in the fusion network

% plot data
% subplot(4,4,13); plot(time_units, magneto_raw, '.m');
% title('Raw Magneto Offset angle'); grid on; set(gca, 'Box', 'off');
% axis_handle_proc(5) = subplot(4,4,14); plot(time_units, head_magneto, '.m');
% title('Absolute magneto angle'); grid on; set(gca, 'Box', 'off');

% combined analysis of input sensor data
% axis_handle_proc(6) = subplot(4,4,[3 16]);
% hold on; plot(time_units, head_gyro, '.r');
% hold on; plot(time_units, head_odometry, '.b');
% hold on; plot(time_units, head_vision, '.g');
% hold on; plot(time_units, head_magneto, '.m');
% grid on; title('Comparison between modalities absolute angles measurements');
% legend('Gyro data', 'Odometry data', 'Vision data', 'Magneto data','Location','NorthWest', 'Location','Best');
% link axes
%linkaxes(axis_handle_raw,'xy');
%linkaxes(axis_handle_proc, 'x');


%% Sensory data analysis for position

%% Vision data analysis

% unit conversion
cm_to_m = 1/100;
% invert axes and negate for alignment with raw data from other modalities
pose_vision(1, :) = -vision_data(:,3)*cm_to_m;
pose_vision(2, :) = -vision_data(:,2)*cm_to_m;

%% TODO GROUND TRUTH DATA FOR FINAL COMPARATIVE ANALYSIS
% invert axes and negate for alignment with raw data from other modalities
ground_truth_y = -pose_vision(1, :);
ground_truth_x = -pose_vision(2, :);
% added smoothing
ground_truth_x = smooth(ground_truth_x, 100);
ground_truth_y = smooth(ground_truth_y, 100);
%%
% accumulate the travelled path from vision data
vision_path = 0;
% integrate path by accumulating displacement
for i = 2:length(vision_data)
    vision_path = vision_path + sqrt((pose_vision(1, i) - pose_vision(1, i-1))^2 + ...
        (pose_vision(2, i) - pose_vision(2, i-1))^2);
end

%% Odometry data analysis

% convert to proper unit
rotpmin_to_radps = 2*pi/60;

% change in pose (derivative) in world refrence frame
dpose_odometry = zeros(3, length(robot_data)); % contains 2D motion components (on x,y,theta)

for i=1:length(robot_data)
    dpose_odometry(:,i) = ...
        ... % rotation matrix
        [cos(ref_sys_rotation(i)), -sin(ref_sys_rotation(i)), 0;...
        sin(ref_sys_rotation(i)),  cos(ref_sys_rotation(i)), 0;...
        0          ,            0             , 1]*...
        ... % kinematics constraints on the wheels
        inv([sqrt(3)/2    ,                        -1/2,     -robot_base;...
        0    ,                           1,     -robot_base;...
        -sqrt(3)/2    ,                        -1/2,     -robot_base])*...
        ...  % wheel radii
        ([radius_t,                  0,                  0;...
        0, radius_t,                  0;...
        0,                  0, radius_t])*...
        ... % wheel angular velocities in rad/s from rot/min
        (rotpmin_to_radps*[robot_data(i,2)/omnirob_buffer_size;...
        robot_data(i,3)/omnirob_buffer_size;...
        robot_data(i,4)/omnirob_buffer_size]);
end

% the pose contains 2 components (X pos, Y pos)
pose_odometry = zeros(2, length(dpose_odometry));
% compute the integrated pose
for i=2:length(robot_data)
    % X pose
    pose_odometry(1, i) = pose_odometry(1, i-1) + ...
        (sample_time*(dpose_odometry(1, i) + dpose_odometry(1, i-1))*.5);
    % Y pose
    pose_odometry(2, i) = pose_odometry(2, i-1) + ...
        (sample_time*(dpose_odometry(2, i) + dpose_odometry(2, i-1))*.5);
end
% compute the integrated path
odometry_path = 0;
for i = 2:length(robot_data)
    odometry_path = odometry_path + ...
        sqrt((pose_odometry(1, i) - pose_odometry(1, i-1))^2 + ...
        (pose_odometry(2, i) - pose_odometry(2, i-1))^2);
end

%% Efferent control signal copy analysis

% efference copy from reference velocities
%---------------------------------------------------------------------------------------------
% Efference copy using the reference velocity set to the motors
wheel_velocity(1,:) = robot_data(:, 27);
wheel_velocity(2,:) = robot_data(:, 28);
wheel_velocity(3,:) = robot_data(:, 29);

% change in pose (derivative) in robot frame and conversion in world frame
dpose_efference = zeros(3, length(robot_data)); % contains all Motion components (x,y,theta)
for i=1:length(robot_data)
    dpose_efference(:,i) = ...
        ... % rotation matrix
        [cos(ref_sys_rotation(i)), -sin(ref_sys_rotation(i)), 0;...
        sin(ref_sys_rotation(i)),  cos(ref_sys_rotation(i)), 0;...
        0          ,            0          , 1]*...
        ... % kinematics constraints on the wheels (rolling)
        inv([sqrt(3)/2    ,                        -1/2,     -robot_base;...
        0    ,                           1,     -robot_base;...
        -sqrt(3)/2    ,                        -1/2,     -robot_base])*...
        ...  % wheel radii
        [radius_t,                  0,                  0;...
        0, radius_t,                  0;...
        0,                  0, radius_t]*...
        ...  % wheel angular velocities in rad/s from rot/min
        (rotpmin_to_radps*[wheel_velocity(1, i);...
        wheel_velocity(2, i);...
        wheel_velocity(3, i)]);
end

% the pose contains 2 components (X, Y)
pose_efference= zeros(2, length(dpose_efference));
% init integration
dpose_efference_ant(1) = dpose_efference(1, 1);
dpose_efference_ant(2) = dpose_efference(2, 1);
for i=2:length(robot_data)
    % X pose
    pose_efference(1, i) = pose_efference(1, i-1) + ...
        (sample_time*(dpose_efference_ant(1) + dpose_efference(1, i-1))*.5);
    % Y pose
    pose_efference(2, i) = pose_efference(2, i-1) + ...
        (sample_time*(dpose_efference_ant(2) + dpose_efference(2, i-1))*.5);
    % update history
    dpose_efference_ant(1) = dpose_efference(1, i-1);
    dpose_efference_ant(2) = dpose_efference(2, i-1);
end
% compute the integrated path
efference_path = 0;
for i = 2:length(robot_data)
    efference_path = efference_path + ...
        sqrt((pose_efference(1, i) - pose_efference(1, i-1))^2 + ...
        (pose_efference(2, i) - pose_efference(2, i-1))^2);
end

%---------------------------------------------------------------------------------------------
% Vision data
x_vision = -pose_vision(2, :);
y_vision = -pose_vision(1, :);
%---------------------------------------------------------------------------------------------
% Odometry data
x_odometry = -pose_odometry(2, :);
y_odometry = -pose_odometry(1, :);
%---------------------------------------------------------------------------------------------
% Efference copy
x_efference = -pose_efference(2, :);
y_efference = -pose_efference(1, :);
%---------------------------------------------------------------------------------------------

%% Sensory data analysis for pose estimation
% vis_fig = figure;
% set(gcf, 'color', 'w');
% subplot(2,3,[1 4]);
% % visualize trajectory of the robot from vision
% plot(-pose_vision(2,:), -pose_vision(1, :), '.g', 'LineWidth', 4); hold on;
% % visualize trajectory of the robot from odometry
% plot(-pose_odometry(2,:), -pose_odometry(1, :), '.b', 'LineWidth', 4); hold on
% % visualize trajectory of the robot from efference copy
% plot(-pose_efference(2, :), -pose_efference(1, :), '.y', 'LineWidth', 4); hold on; grid on;box off; grid on;
% title('Robot trajectory');legend(sprintf('Vision integrated path: %f',vision_path),...
%     sprintf('Odometry integrated path: %f', odometry_path),...
%     sprintf('Motor efference copy integrated path: %f', efference_path));box off; grid on;

% plot separate Motion components
% subplot(2,3,2);
% plot(time_units, -pose_vision(2, :), '.r','LineWidth', 4); hold on;
% plot(time_units, -pose_vision(1, :), '.b','LineWidth', 4); box off; grid on;
% title(sprintf('Motion components from vision - Integrated path: %f', vision_path));
% legend('X axis component', 'Y axis component');
% subplot(2,3,3);
% plot(time_units, -pose_odometry(2, :), '.r','LineWidth', 4); hold on;
% plot(time_units, -pose_odometry(1, :), '.b','LineWidth', 4); hold on;
% plot(time_units, -pose_vision(2, :), '--k','LineWidth', 3); hold on;
% plot(time_units, -pose_vision(1, :), ':k','LineWidth', 3); box off; grid on;
% title(sprintf('Motion components from odometry - Integrated path: %f', odometry_path));
% legend('X axis component', 'Y axis component', 'X vision', 'Y vision');
% subplot(2,3,5);
% plot(time_units, -pose_efference(2, :), '.r','LineWidth', 4); hold on;
% plot(time_units, -pose_efference(1, :), '.b','LineWidth', 4); hold on;
% plot(time_units, -pose_vision(2, :), '--k','LineWidth', 3); hold on;
% plot(time_units, -pose_vision(1, :), ':k','LineWidth', 3); box off; grid on;
% title(sprintf('Motion components from efference - Integrated path: %f', efference_path));
% legend('X axis component', 'Y axis component', 'X vision', 'Y vision');
% close all;
% data struct to return with both motion components
% time units
data.timeunits = time_units;
% heading angle
data.heading.gyro = head_gyro;
data.heading.magneto = head_magneto;
data.heading.odometry = head_odometry;
data.heading.cam = head_vision;
% position
data.pose.cam = -pose_vision;
data.pose.odometry = -pose_odometry;
data.pose.efference = -pose_efference;
end