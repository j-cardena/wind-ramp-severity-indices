function f = get_demand_factor(hour, cfg)
%GET_DEMAND_FACTOR Demand factor by hour (inclusive windows).
h = double(hour);
if cfg.peak.evening(1) <= h && h <= cfg.peak.evening(2)
    f = 1.5;
elseif cfg.peak.morning(1) <= h && h <= cfg.peak.morning(2)
    f = 1.3;
elseif cfg.peak.midday(1) <= h && h <= cfg.peak.midday(2)
    f = 1.1;
else
    f = 0.8;
end
end
