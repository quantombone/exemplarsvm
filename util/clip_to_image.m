function boxes = clip_to_image(boxes, imbb)
% Clip boxes to image dimensions
% boxes = clip_to_image(boxes, imbb)
% imbb is a 4-vector which is generally [1 1 size(I,2) size(I,1)]
%

if size(boxes,1) == 0
  return;
end

boxes(:,1) = cap_range(boxes(:,1),imbb([1 3]));
boxes(:,3) = cap_range(boxes(:,3),imbb([1 3]));

boxes(:,2) = cap_range(boxes(:,2),imbb([2 4]));
boxes(:,4) = cap_range(boxes(:,4),imbb([2 4]));
