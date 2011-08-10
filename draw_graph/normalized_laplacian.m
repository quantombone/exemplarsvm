function L = normalized_laplacian(A)
A = A - diag(diag(A));
degs = sum(A,1);
Tih = spdiag(degs.^-.5);
L = speye(size(A)) - Tih*A*Tih;

%% enforce machine precision un-symmetry
%% helps eigs make sure we don't get negative eigenvalues
L = (L + L')/2;
