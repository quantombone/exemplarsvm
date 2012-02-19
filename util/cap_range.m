function values_capped = cap_range(values,min_value,max_value)
% Cap the values to the range [min_value,max_value]
% [values_capped] = cap_range(values,min_value,max_value);
% [values_capped] = cap_range(values,[min_value max_value]);
%
if ~exist('max_value','var')
  max_value = min_value(2);
  min_value = min_value(1);
end
values_capped = min(values,max_value);
values_capped = max(values_capped,min_value);
