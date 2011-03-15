function res = max_or_this(x,value)
if length(x) > 0
  res = max(x);
  return;
end
res = value;