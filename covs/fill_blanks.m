function x = fill_blanks(x,res,res2)

if ~exist('res2','var')
  res2 = subcov(res,size(x));
end
ids = find(repmat(sum(x.^2,3)==0,[1 1 31]));

if length(ids) == 0
  return;
end
otherids = find(repmat(sum(x.^2,3)~=0,[1 1 31]));

mu1 = res2.mean(ids);
mu2 = res2.mean(otherids);
a = x(otherids);
cov12 = res2.c(ids,otherids);
cov22 = res2.c(otherids,otherids);

x2 = mu1 + cov12*pinv(cov22)*(a-mu2);
x(ids) = x2;

