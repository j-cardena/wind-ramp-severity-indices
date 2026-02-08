function [results, cfg] = calculate_batch(ramps, cfg)
%CALCULATE_BATCH Two-pass batch calculation (compute norm bounds then indices).
% ramps: struct array of ramp events

raw = struct('RAI',[],'RSCI',[],'OSI',[],'GIP',[]);
for i = 1:numel(ramps)
    r = ramps(i);
    raw.RAI(end+1)  = calculate_rai(r); %#ok<AGROW>
    raw.RSCI(end+1) = calculate_rsci(r); %#ok<AGROW>
    raw.OSI(end+1)  = calculate_osi(r, cfg); %#ok<AGROW>
    raw.GIP(end+1)  = calculate_gip(r, cfg); %#ok<AGROW>
end

% Update bounds
cfg.norm_bounds.RAI  = [min(raw.RAI)  max(raw.RAI)];
cfg.norm_bounds.RSCI = [min(raw.RSCI) max(raw.RSCI)];
cfg.norm_bounds.OSI  = [min(raw.OSI)  max(raw.OSI)];
cfg.norm_bounds.GIP  = [min(raw.GIP)  max(raw.GIP)];

% Second pass: compute full results with normalization
results = repmat(struct(), numel(ramps), 1);
for i = 1:numel(ramps)
    results(i) = calculate_all_indices(ramps(i), cfg, true);
end
end
