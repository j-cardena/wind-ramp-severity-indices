function v = calculate_osi(ramp, cfg)
%CALCULATE_OSI Operational Stress Index (asymmetric up/down).
base_stress = ramp.magnitude * (1 + ramp.rate);

if strcmpi(ramp.direction,'down')
    R_reserve = exp(-3 * ramp.end_power);
else
    R_reserve = exp(-3 * (1 - ramp.end_power));
end
reserve_factor = 1 + R_reserve;

ramp_reserve = 1 + ramp.rate ^ 1.5;
demand_factor = get_demand_factor(ramp.start_time, cfg);

v = base_stress * reserve_factor * ramp_reserve * demand_factor;
end
