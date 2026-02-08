function ramps = load_sample_ramps(csvPath)
%LOAD_SAMPLE_RAMPS Load ramps from the repository sample CSV.
% Expects columns: ramp_id,start_hour,direction,power_series

if nargin < 1 || isempty(csvPath)
    csvPath = fullfile('..','data','sample_ramps.csv');
end

T = readtable(csvPath, 'TextType','string');

ramps = repmat(struct(), height(T), 1);
for i = 1:height(T)
    p = utils.parse_power_series(T.power_series(i));
    ts = 0:(numel(p)-1);
    ramps(i) = make_ramp_event(p, T.start_hour(i), T.direction(i), ts);
    ramps(i).ramp_id = T.ramp_id(i);
end
end
