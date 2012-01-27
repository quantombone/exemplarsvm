function z = distSqr_fast(x,y,x2,y2) %x2 = sum(x.^2,1)'; %y2 = sum(y.^2,1);
% function z = distSqr_fast(x,y,x2,y2)
%
% Return matrix of all-pairs squared distances between the vectors
% in the columns of x and y.
%
% INPUTS
% 	x 	dxn matrix of vectors
% 	y 	dxm matrix of vectors
%
% OUTPUTS
% 	z 	nxm matrix of squared distances
%
% This routine is faster when m<n than when m>n. In other words y should be
% smaller than x.
%
% David Martin <dmartin@eecs.berkeley.edu>
% March 2003

% Based on dist2.m code,
% Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

if ~exist('y','var')
  y = x;
end

if ~exist('x2','var')
  x2 = sum(x.^2,1)';
end

if ~exist('y2','var')
  y2 = sum(y.^2,1);
end

if size(x,1) ~= size(y,1), 
  error('size(x,1)~=size(y,1)'); 
end

[d,n] = size(x);
[d,m] = size(y);

%z = x'*y
%z = repmat(x2,1,m) ...
%    + repmat(y2,n,1) ...
%    - 2*x'*y;
%return

z = x'*y;

for i = 1:m,
  z(:,i) = x2 + y2(i) - 2*z(:,i);
end

%z = zeros(n,m);
%for i = 1:m,
%  z(:,i) = x2 + y2(i) - 2*x'*y(:,i);
%end
