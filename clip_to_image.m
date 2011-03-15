function boxes = clip_to_image(boxes,imbb)
%clip boxes to image (just changes the max dimensions)
if size(boxes,1) == 0
  return;
end

for i = 1:2
  boxes(:,i) = max(imbb(i),boxes(:,i));
end

for i = 3:4
  boxes(:,i) = min(imbb(i),boxes(:,i));
end
