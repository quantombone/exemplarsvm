function result = ecat(x,defaultdim)
if nargin == 1
  defaultdim = 2;
end
result = cat(2,x{:});
