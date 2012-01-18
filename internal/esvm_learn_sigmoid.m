function beta = esvm_learn_sigmoid(scores, os)
% function beta = esvm_learn_sigmoid(scores, os)
% Fit a sigmoid to the scores verus os. Examples with high scores
% and high os's will "fit" well.  This is a soft way of counting
% the number of "good" detections before the first bad one.
%
% Instances with os>.5 are treated as positives, instances with os
% between [.2 and .5] are treated as dont-cares, and os<.2 is treated
% as negative.  The parameters have been tweaked such that raw SVM
% output scores which assign a 1.0 for the positive and scores around
% -.9 for negatives, produce reasonable fits.  If the scores do not
% satisfy this distribution, then there might be a local minimum
% problem -- check this first if fitting is not working for you)
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

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
% The function we are using (named robust, but doesn't have to be
% robust as can be seen from the code below)
% Robust fitting doesn't seem to help much
% r = mean(min(sigma,d.^2));
r = mean(d.^2);
