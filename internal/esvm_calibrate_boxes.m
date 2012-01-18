function bboxes = esvm_calibrate_boxes(bboxes, betas)
% Apply learned Platt-calibration parameters onto raw detection
% scores in bboxes
%
% Given a matrix of object detection bounding boxes (which contain exemplar
% ids as element 6), rescale them using the learned betas
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if size(bboxes,1) == 0
  return;
end

ids = bboxes(:,6);
if ~exist('betas','var') || length(betas)==0
  betas(ids,1) = 1;
  betas(ids,2) = 0;
end
scores = betas(ids,1).*(bboxes(:,end) - betas(ids,2));
%bboxes(:,end) = scores;
bboxes(:,end) = 1./(1+exp(-scores));
