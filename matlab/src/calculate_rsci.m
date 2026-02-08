function v = calculate_rsci(ramp)
%CALCULATE_RSCI Ramp Shape Complexity Index.
% RSCI = (L_actual / L_straight) * (1 + N_inflection / n)
if ramp.n_points < 2
    v = 1.0; return;
end
n = ramp.n_points;
rates = diff(ramp.power);
path_segments = sqrt(1 + rates.^2);
L_actual = sum(path_segments);
L_straight = sqrt((n - 1)^2 + ramp.magnitude^2);
if L_straight == 0
    v = 1.0; return;
end

% Inflection count (direction changes), ignoring zeros
if numel(rates) < 2
    n_inflections = 0;
else
    signs = sign(rates);
    signs = signs(signs ~= 0);
    if numel(signs) < 2
        n_inflections = 0;
    else
        n_inflections = sum(diff(signs) ~= 0);
    end
end

path_ratio = L_actual / L_straight;
inflection_factor = 1 + n_inflections / n;
v = path_ratio * inflection_factor;
end
