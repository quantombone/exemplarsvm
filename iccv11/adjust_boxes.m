function top = adjust_boxes(boxes, models)
%Here we take the detection boxes and adjust them so they are 'GT'
%boxes.

%This step is necessary since the exemplars are framed in a slightly
%different window (one of the coarse aspect ratios based on the 8
%pixel cells) than the actual GT window (which can have any possible
%aspect ratio up to the resolution of the original image)

%% Each detection is an alignment between the "coarse_box" and a
%detection window "d", once we find the rigid transformation
%between "coarse_box" and "d", we can apply the same projection to
%the ground truth window

%% Tomasz Malisiewicz (tomasz@cmu.edu)

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
  c = models{boxes(i,6)}.model.coarse_box; %(1,:);
  gt = models{boxes(i,6)}.gt_box;
  
  %find the xform from c to d
  xform = find_xform(c, d(1:4));
  top(i,1:4) = apply_xform(gt, xform);      
end
