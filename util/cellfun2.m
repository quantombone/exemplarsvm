function res = cellfun2(funer, structy)
%An alias for cellfun with UniformOutput set to false
if nargin<2
  error('Not enough arguments to cellfun2\n');
end
res = cellfun(funer, structy, 'UniformOutput', false);