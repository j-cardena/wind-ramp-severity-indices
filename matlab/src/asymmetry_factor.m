function f = asymmetry_factor(ramp, cfg)
%ASYMMETRY_FACTOR Asymmetry factor for GIP.
if strcmpi(ramp.direction,'down')
    if ramp.end_power < 0.3 && is_peak_hour(ramp.start_time, cfg)
        f = 1.5;
    else
        f = 1.2;
    end
else
    if ramp.end_power > 0.7 && is_peak_hour(ramp.start_time, cfg)
        f = 1.3;
    else
        f = 1.0;
    end
end
end
