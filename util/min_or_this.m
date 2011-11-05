function [res,ind] = min_or_this(x,value)
if length(x) > 0
  [res,ind] = min(x);
  return;
end
res = value;
ind = -1;