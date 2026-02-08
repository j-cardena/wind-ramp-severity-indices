function v = calculate_gip(ramp, cfg)
%CALCULATE_GIP Grid Impact Potential.

mag_term = sqrt(ramp.magnitude);
if ramp.rate > 0
    rate_term = sqrt(ramp.rate);
else
    rate_term = 0;
end
dur_mod = duration_modifier(ramp.duration);
f1 = mag_term * rate_term * dur_mod;

avg_power = (ramp.start_power + ramp.end_power) / 2;
extreme_penalty = 1 + 4 * (avg_power - 0.5) ^ 2;
time_factor = get_time_factor(ramp.start_time);
f2 = extreme_penalty * time_factor;

f3 = asymmetry_factor(ramp, cfg);

v = f1 * f2 * f3;
end
