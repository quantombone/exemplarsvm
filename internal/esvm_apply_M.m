function boxes = esvm_apply_M(xraw, boxes, M)
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

boxes(:,end) = esvm_apply_M_driver(xraw, boxes, M);

return;

%uims = unique(boxes(:,11));
%for i = 1:length(xraw)
  % hits = find(boxes(:,11)==uims(i));
  % b = boxes(hits,:);
  % b(:,end) = b(:,end)+1;
  
  % if isfield(model,'M')
  %   len = length(model.models);
  %   gt = model.M.neighbor_thresh;
  %   curM = model.M;
  % else
  %   len = length(model.w)/2;
  %   gt = model.neighbor_thresh;
  %   curM = model;
  % end
  
  % [xraw,nbrlist{i}] = esvm_get_M_features(b,len, ...
  %                                         gt);
  r2 = esvm_apply_M_driver(xraw,boxes,M);
  boxes(:,end) = r2;
%end


function r = esvm_apply_M_driver(x, boxes, M)

if prod(size(x))==0
  r = zeros(1,0);
  return;
end
exids = boxes(:,6);

%Because the co-occurrence matrix treats exemplar flips as separate
%exemplars, we need to check if an exemplar has been flipped at
%update its exemplar index
exids(boxes(:,7)==1) = exids(boxes(:,7)==1) + size(x,1)/2;
r = zeros(size(boxes,1),1);

if length(M.w) == 1
  r = M.w{1}'*x;
  r = r';
  return;
end

for i = 1:size(boxes,1)  
  r(i) = (M.w{exids(i)}'*x(:,i));
end
