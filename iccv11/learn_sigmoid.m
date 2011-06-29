function beta = learn_sigmoid(scores,os)
%Learn a sigmoid to the stuff to fit the scores to the os, os is
%assumed to be from within-class examples.  Instances with os>.5
%are treated as positives, instances with os between [.2 and .5]
%are treated as dont-cares, and os<.2 is treated as negative

x = scores;
y = os;

%capping to one, (DO THIS OR NOT? YES)
y(y>=.5) = 1;

y(y<=.2) = 0;
bads = y>.2 & y<.5;
y(bads) = [];
x(bads) = [];

reg_constant = .000001;
sigma = 1;
fun = @(beta)robust_loss(1./(1+exp(-beta(1)*(x-beta(2))))-y,sigma)+ ...
    reg_constant*beta(1).^2;

guess2 = 100;
if sum(y>.5) > 0
  guess2 = mean(x(y>.5));
end

beta = [3.0 guess2];
beta = fminsearch(fun, beta,...
                  optimset('MaxIter',10000,...
                           'MaxFunEvals',10000,...
                           'Display','off'));

function r = robust_loss(d,sigma)
%r = mean(min(sigma,d.^2));
r = mean(d.^2);
