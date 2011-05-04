function bbox2=flip_box(bbox,sizeI)
%Flip the box using a L-R reflection
if size(bbox,1) == 0
  bbox2 = bbox;
  return;
end

W = bbox(3) - bbox(1) + 1;
H = bbox(4) - bbox(2) + 1;

bbox2 = bbox;
bbox2(3) = sizeI(2)-bbox2(1);
bbox2(1) = bbox2(3)-W+1;

return;

W2 = bbox2(3) - bbox2(1) + 1;
H2 = bbox2(4) - bbox2(2) + 1;

W/H - W2/H2

figure(1)
clf
imagesc(zeros(sizeI(1),sizeI(2),3))
plot_bbox(bbox)
plot_bbox(bbox2,'',[1 0 0])

