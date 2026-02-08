function f = get_time_factor(hour)
%GET_TIME_FACTOR Time factor for GIP. Matches Python implementation.
h = double(hour);
if 7 <= h && h <= 21
    f = 0.7 + 0.3 * sin(pi * (h - 7) / 14);
else
    f = 0.5;
end
end
