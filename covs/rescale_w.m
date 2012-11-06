function [w,b] = rescale_w(w,meanpos,meanneg)
%rescale decision boundary such that when applying rule: f(x) =
%w'*x-b, scores of +1 map to positive mean and -g map to negative
%mean

[alphas] = inv([w(:)'*meanpos 1; w(:)'*meanneg 1])*[1 -1]';
w = w*alphas(1);
b = -alphas(2);
