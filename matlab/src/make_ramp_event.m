function ramp = make_ramp_event(power, start_hour, direction, timestamps)
%MAKE_RAMP_EVENT Create a ramp event struct.
%
% Fields mirror the Python RampEvent dataclass:
% - power (row vector)
% - timestamps (row vector, hours from start)
% - start_time (int 0-23)
% - direction ('up'|'down')

if nargin < 4 || isempty(timestamps)
    timestamps = 0:(numel(power)-1);
end

ramp = struct();
ramp.power = power(:)'; % row
ramp.timestamps = timestamps(:)'; % row
ramp.start_time = int32(start_hour);
ramp.direction = char(direction);

% Derived values
ramp.magnitude = abs(ramp.power(end) - ramp.power(1));
ramp.duration  = ramp.timestamps(end) - ramp.timestamps(1);
if ramp.duration == 0
    ramp.rate = 0.0;
else
    ramp.rate = ramp.magnitude / ramp.duration;
end
ramp.start_power = ramp.power(1);
ramp.end_power   = ramp.power(end);
ramp.n_points    = numel(ramp.power);
end
