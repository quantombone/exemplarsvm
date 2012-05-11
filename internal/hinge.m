function y = hinge(x)
%return the hinge squared function
y = max(1-x,0).^2;