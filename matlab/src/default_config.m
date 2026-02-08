function cfg = default_config()
%DEFAULT_CONFIG Default weights and peak hour definitions.

cfg.weights = struct('RAI',0.452,'RSCI',0.271,'OSI',0.107,'GIP',0.170);

% Peak hour windows (inclusive)
cfg.peak.evening = [17 21];
cfg.peak.morning = [7 9];
cfg.peak.midday  = [12 14];

% Normalization bounds placeholders (min,max). Updated in batch processing.
cfg.norm_bounds = struct( ...
    'RAI',[0 1], ...
    'RSCI',[1 2], ...
    'OSI',[0 1], ...
    'GIP',[0 1] ...
);
end
