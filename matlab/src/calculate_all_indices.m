function out = calculate_all_indices(ramp, cfg, doNormalize)
%CALCULATE_ALL_INDICES Compute all indices and basic parameters.
if nargin < 3
    doNormalize = true;
end

rai = calculate_rai(ramp);
rsci = calculate_rsci(ramp);
osi = calculate_osi(ramp, cfg);
gip = calculate_gip(ramp, cfg);

rai_n = rai; rsci_n = rsci; osi_n = osi; gip_n = gip;
if doNormalize
    rai_n  = normalize_value(rai,  cfg.norm_bounds.RAI);
    rsci_n = normalize_value(rsci, cfg.norm_bounds.RSCI);
    osi_n  = normalize_value(osi,  cfg.norm_bounds.OSI);
    gip_n  = normalize_value(gip,  cfg.norm_bounds.GIP);
end

ecsi = cfg.weights.RAI * rai_n + cfg.weights.RSCI * rsci_n + ...
       cfg.weights.OSI * osi_n + cfg.weights.GIP * gip_n;

out = struct();
out.magnitude = ramp.magnitude;
out.duration  = ramp.duration;
out.rate      = ramp.rate;
out.direction = ramp.direction;
out.start_time = double(ramp.start_time);

out.RAI = rai;
out.RSCI = rsci;
out.OSI = osi;
out.GIP = gip;
out.ECSI = ecsi;
end
