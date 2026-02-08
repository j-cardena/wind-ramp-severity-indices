% RUN_ALL Compute severity indices (MATLAB) and export results.
% Run from repo root:
%   cd matlab
%   run('scripts/run_all.m')

here = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(here);

addpath(fullfile(matlabRoot,'src'));
addpath(matlabRoot); % for +utils

cfg = default_config();

% Load sample ramps from repo data/
ramps = load_sample_ramps(fullfile(matlabRoot,'..','data','sample_ramps.csv'));

% Compute batch results
[results, cfg] = calculate_batch(ramps, cfg);

% Convert to table for export
n = numel(results);
ramp_id = zeros(n,1);
direction = strings(n,1);
start_time = zeros(n,1);
magnitude = zeros(n,1);
duration = zeros(n,1);
rate = zeros(n,1);
RAI = zeros(n,1);
RSCI = zeros(n,1);
OSI = zeros(n,1);
GIP = zeros(n,1);
ECSI = zeros(n,1);
severity = strings(n,1);

for i = 1:n
    ramp_id(i) = ramps(i).ramp_id;
    direction(i) = string(results(i).direction);
    start_time(i) = results(i).start_time;
    magnitude(i) = results(i).magnitude;
    duration(i) = results(i).duration;
    rate(i) = results(i).rate;
    RAI(i) = results(i).RAI;
    RSCI(i) = results(i).RSCI;
    OSI(i) = results(i).OSI;
    GIP(i) = results(i).GIP;
    ECSI(i) = results(i).ECSI;
    severity(i) = string(classify_severity(ECSI(i)));
end

outT = table(ramp_id, start_time, direction, magnitude, duration, rate, ...
    RAI, RSCI, OSI, GIP, ECSI, severity);

% Ensure results directories exist
resultsDir = fullfile(matlabRoot,'..','results');
figDir = fullfile(resultsDir,'figures');
if ~exist(resultsDir,'dir'); mkdir(resultsDir); end
if ~exist(figDir,'dir'); mkdir(figDir); end

% Write CSV
outCsv = fullfile(resultsDir,'matlab_ramp_indices.csv');
writetable(outT, outCsv);

% Basic plots
% 1) ECSI by direction (box chart)
f1 = figure('Visible','off');
boxchart(categorical(direction), ECSI);
xlabel('direction'); ylabel('ECSI');
title('ECSI by direction (MATLAB)');
saveas(f1, fullfile(figDir,'matlab_ecsi_boxplot.png'));

% 2) Histogram
f2 = figure('Visible','off');
histogram(ECSI);
xlabel('ECSI'); ylabel('count');
title('ECSI distribution (MATLAB)');
saveas(f2, fullfile(figDir,'matlab_ecsi_hist.png'));

close(f1); close(f2);

fprintf('Wrote: %s\n', outCsv);
fprintf('Saved figures to: %s\n', figDir);
