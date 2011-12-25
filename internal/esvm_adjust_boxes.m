function top = esvm_adjust_boxes(boxes, models)
%Adjust coarse-frame detections into ground-truth frames.
%function top = esvm_adjust_boxes(boxes, models)
%
%This step is necessary since the exemplars are framed in a slightly
%different window (one of the coarse aspect ratios based on the 8
%pixel cells) than the actual GT window (which can have any possible
%aspect ratio up to the resolution of the original image)

%% Each detection is an alignment between the "coarse_box" and a
%detection window "d", once we find the rigid transformation
%between "coarse_box" and "d", we can apply the same projection to
%the ground truth window
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).


top = boxes;

if strcmp(models{1}.models_name,'dalal') || ...
      ~isfield(models{1},'gt_box')
  return;
end

if numel(boxes)==0
  return;
end

top(:,1:4) = 0;

for i = 1:size(boxes,1)
  d = boxes(i,:);
  c = models{boxes(i,6)}.model.bb(1,1:4); %(1,:);
  gt = models{boxes(i,6)}.gt_box;
  
  %find the xform from c to d
  xform = find_xform(c, d(1:4));
  top(i,1:4) = apply_xform(gt, xform);      
end
