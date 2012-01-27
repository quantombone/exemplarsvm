function result = emap(func, x)
%An alias for cellfun, which applies function func to all elements
%of cell array x, but with UniformOutput set to false
if nargin<2
  error('Not enough arguments to cellfunc2\n');
end
result = cellfun(func, x, 'UniformOutput', false);