function r = esvm_apply_M(x, boxes, M)
% Apply boosting "co-occurrence" matrix M to the boxes
% function r = esvm_apply_M(x, boxes, M)
%
% Apply the multiplexer matrix M which boosts the scores of a
% window based on its friends and their scores embedded in the
% context feature vector x
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if prod(size(x))==0
  r = zeros(1,0);
  return;
end
exids = boxes(:,6);

%Because the co-occurrence matrix treats exemplar flips as separate
%exemplars, we need to check if an exemplar has been flipped at
%update its exemplar index
exids(boxes(:,7)==1) = exids(boxes(:,7)==1) + size(x,1)/2;
r = zeros(1,size(boxes,1));

for i = 1:size(boxes,1)
  r(i) = (M.w{exids(i)}'*x(:,i) + sum(x(:,i)))-M.b{exids(i)};
end
