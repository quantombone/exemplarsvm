function [v,d,mu] = mypca(xxx,K)
%perform super PCA and return the eigenvectors as columns, the
%eigenvalues, and the mean vector

%Tomasz Malisiewcz (tomasz@cmu.edu)
[d,n] = size(xxx);
if d > 4000
  fprintf(1,'too big!\n');
  v = [];
  mu = [];
  return;
end

mu = mean(xxx,2);
c = xxx*xxx';
c = c - n*mu*mu';
%c = cov(xxx');
%symmetrize to eliminate some issues
c = (c+c')/2;

if exist('K','var')
  K = min(min(K,size(xxx,2)),size(xxx,1));
end

if exist('K','var')
  [v,d] = eigs(c,K);
else
  [v,d] = eig(c);
end


d = diag(d);
d = max(d,0.0);
%d = d / sum(d);
[aa,bb] = sort(d,'descend');
v = v(:,bb);
d = d(bb);
