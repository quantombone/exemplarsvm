function [res,ind] = max_or_this(x,value)
if length(x) > 0
  [res,ind] = max(x);
  return;
end
res = value;
ind = -1;