function p = parse_power_series(seriesStr)
%PARSE_POWER_SERIES Parse a Python-style list string like "[0.1, 0.2, 0.3]".
% Returns a row vector double.

if isstring(seriesStr) || ischar(seriesStr)
    s = char(seriesStr);
else
    error('seriesStr must be string or char.');
end

s = strtrim(s);

% Remove brackets
s = regexprep(s,'^\[','');
s = regexprep(s,'\]$','');

if isempty(strtrim(s))
    p = [];
    return;
end

% Split by commas
parts = regexp(s,',','split');
p = zeros(1,numel(parts));
for i = 1:numel(parts)
    p(i) = str2double(strtrim(parts{i}));
end
end
