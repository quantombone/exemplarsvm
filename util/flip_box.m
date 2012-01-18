function bbox2=flip_box(bbox,sizeI)
%Flip a matrix of boxes using a L-R reflection
%Each row is a new BB with the first four columns as the BB location
if size(bbox,1) == 0
  bbox2 = bbox;
  return;
end

W = bbox(:,3) - bbox(:,1) + 1;
H = bbox(:,4) - bbox(:,2) + 1;

bbox2 = bbox;
bbox2(:,3) = sizeI(2)-bbox2(:,1);
bbox2(:,1) = bbox2(:,3)-W+1;
