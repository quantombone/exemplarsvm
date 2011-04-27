function K = learnKernel(n,D,lambda)

% LEARNKERNEL
% function K = learnKernel(n,D,lambda)
%
% Jonathan Huang

Dp = totriplets(D);
m = size(Dp,1);

cvx_begin
	variable K(n,n) symmetric;
	variable s(m); % slack variables
	minimize sum(s)+lambda*trace(K);
	subject to
		K == semidefinite(n);
		sum(sum(K)) == 0;
		s >= 0;
		for idx=1:m
			ref = Dp(idx,1); qclose = Dp(idx,2); qfar = Dp(idx,3);
			2*K(qclose,ref)-2*K(qfar,ref)+K(qfar,qfar)-K(qclose,qclose)+s(idx)>=1;
		end
cvx_end



function Dp = totriplets(D)
 
m = size(D,1);
k = size(D,2)-1;

Dp = [];
for idx = 1:m
	tmp = [repmat(D(idx,1:2),k-1,1) D(idx,3:end)'];
	Dp = [Dp; tmp];
end

