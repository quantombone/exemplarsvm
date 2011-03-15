function w = x2w(x, A)
%% Map a HOG template x to a classifier-space template W by applying
%% the per-cell regressor.
%% A: optional ridge matrix (loads from file if not provided)

%Tomasz Malisiewicz (tomasz@cmu.edu)

if ~exist('A','var')
  A = perform_x2w_ridge_regression;
end

xsize = size(x);
x = reshape(x, [], 31)';
x(end+1,:) = 1;
w = A*x;
w = w';
w = reshape(w,xsize);
