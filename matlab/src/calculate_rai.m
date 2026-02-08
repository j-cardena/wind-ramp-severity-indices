function v = calculate_rai(ramp)
%CALCULATE_RAI Ramp Acceleration Index.
% RAI = max|r_{i+1} - r_i| / std(r)
if ramp.n_points < 3
    v = 0.0; return;
end
rates = diff(ramp.power);
if numel(rates) < 2
    v = 0.0; return;
end
accel = diff(rates);
max_accel = max(abs(accel));
rate_std = std(rates, 0); % population std like numpy default (ddof=0)
if rate_std == 0
    v = 0.0; return;
end
v = max_accel / rate_std;
end
