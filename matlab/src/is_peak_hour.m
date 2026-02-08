function tf = is_peak_hour(hour, cfg)
%IS_PEAK_HOUR True if within morning or evening peak windows.
h = double(hour);
tf = (cfg.peak.evening(1) <= h && h <= cfg.peak.evening(2)) || ...
     (cfg.peak.morning(1) <= h && h <= cfg.peak.morning(2));
end
