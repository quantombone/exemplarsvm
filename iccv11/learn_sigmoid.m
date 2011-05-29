function beta = learn_sigmoid(scores,os)
%Learn a sigmoid to the stuff
%%function process_grids(grid)

x = scores;
y = os;

%DO THIS OR NOT?
y(y>=.5) = 1;

y(y<=.2) = 0;
bads = y>.2 & y<.5;
y(bads) = [];
x(bads) = [];

reg_constant = .000001;
sigma = 1;
fun = @(beta)robust_loss(1./(1+exp(-beta(1)*(x-beta(2))))-y,sigma)+ ...
    reg_constant*beta(1).^2;

%beta = [3.0 -.5];

guess2 = 100;
if sum(y>.5) > 0
  guess2 = mean(x(y>.5));
end

beta = [3.0 guess2];
%fprintf(1,'Start Values is %.3f\n',fun(beta));
beta = fminsearch(fun,beta,optimset('MaxIter',10000,'MaxFunEvals',10000));
%fprintf(1,'End Value is %.3f\n',fun(beta));

function r = robust_loss(d,sigma)
%r = mean(min(sigma,d.^2));
r = mean(d.^2);
