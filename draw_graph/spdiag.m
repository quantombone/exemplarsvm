function W=spdiag(D);
% Timothee Cour, 04-Aug-2008 20:46:38 -- DO NOT DISTRIBUTE

D=D(:);
n=length(D);
W = spdiags(D,0,n,n);


