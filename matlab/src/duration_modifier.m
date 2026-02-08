function f = duration_modifier(duration)
%DURATION_MODIFIER Penalize very short or very long ramps.
d = double(duration);
if d < 3
    f = 0.7 + 0.1 * d;
elseif d > 15
    f = 1.0 - 0.02 * (d - 15);
else
    f = 1.0;
end
end
