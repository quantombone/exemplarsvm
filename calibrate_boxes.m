function bboxes = calibrate_boxes(bboxes, betas)
%Given a matrix of object detection boxes (which contain exemplar
%ids), rescale them using the learned betas

ids = bboxes(:,6);
if ~exist('betas','var') | length(betas)==0
  betas(ids,1) = 1;
  betas(ids,2) = 0;
end
scores = betas(ids,1).*(bboxes(:,end) - betas(ids,2));
%bboxes(:,end) = scores;
bboxes(:,end) = 1./(1+exp(-scores));