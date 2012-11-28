function data_set2 = construct_scene_dataset(data_set)
% Construct a dataset from a sequence of images, where a
% tight-fitting bounding box is placed inside each image as the only
% annotation

% Tomasz Malisiewicz (tomasz@csail.mit.edu)
%

inds = 1:length(data_set);
for iii = 1:length(inds)
  i = inds(iii);
  I = data_set{i};
  clear s objects
  s.I = I;
  I = toI(I);
  objects(1).class = sprintf('%d',iii);
  objects(1).truncated = 0;
  objects(1).difficult = 0;
  W = size(I,2)*.1;
  H = size(I,1)*.1;
  objects(1).bbox = [1+W 1+H size(I,2)-W size(I,1)-H];
  s.objects = objects;
  data_set2{iii} = s;
end

