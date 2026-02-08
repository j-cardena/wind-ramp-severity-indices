function cls = classify_severity(ecsi)
%CLASSIFY_SEVERITY Severity class based on ECSI (0-1).
if ecsi < 0.25
    cls = 'low';
elseif ecsi < 0.50
    cls = 'moderate';
elseif ecsi < 0.75
    cls = 'high';
else
    cls = 'critical';
end
end
