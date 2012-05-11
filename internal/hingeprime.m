function y = hingeprime(x)
%hingeprime function

%squared hinge loss derivative
y = -2*(1-x);

%linear hinge loss derivative
%y = x*0-1;

y(x>=1) = 0;