function y = normalize_value(x, bounds)
%NORMALIZE_VALUE Min-max normalize to [0,1] with clipping.
minv = bounds(1); maxv = bounds(2);
if maxv == minv
    y = 0.5;
else
    y = (x - minv) / (maxv - minv);
    y = max(0, min(1, y));
end
end
