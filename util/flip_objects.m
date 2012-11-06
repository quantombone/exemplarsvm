function [I,objects] = flip_objects(I,objects)
I = flip_image(I);
for i = 1:length(objects)
  objects(i).bbox = flip_bbox(objects(i).bbox,size(I));
  
  objects(i).polygon = flip_polygon(objects(i).polygon,size(I));
  
end

function p = flip_polygon(p,sizeI)
W = bbox(:,3) - bbox(:,1) + 1;
H = bbox(:,4) - bbox(:,2) + 1;

bbox2 = bbox;
bbox2(:,3) = sizeI(2)-bbox2(:,1);
bbox2(:,1) = bbox2(:,3)-W+1;
