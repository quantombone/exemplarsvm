function [evals,evecs] = learn_cov(res)
% Compute eigenvalues and eigenvectors of the HOG covariance matrix
% estimated from the fields of the input [res].  If 3 outputs are
% specified, then returns a stack of eigenvector images by performing
% eigendecomposition of the covariance matrix

c = res.c;

%c = inv(c);
tic
[v,d] = eig(c);
d = diag(d);
[aa,bb] = sort(d,'descend');
v = v(:,bb);
d = d(bb);
toc

evals = d;
evecs = v;
