function data_set2 = construct_bb_dataset(data_set, bb)

% Construct a dataset from a sequence of images, and a set of
% bounding boxes

% Tomasz Malisiewicz (tomasz@csail.mit.edu)
%

inds = 1:length(data_set);
c = 1;
for iii = 1:length(inds)
  i = inds(iii);
  
  hits = find(bb(:,11)==i);
  if length(hits) == 0
    continue
  end
  
  I = data_set{i};
  clear s objects
  s.I = I.I;
  I = toI(I);
  for j = 1:length(hits)
    objects(j).class = sprintf('det');
    objects(j).truncated = 0;
    objects(j).difficult = 0;
    objects(j).bbox = bb(hits(j),1:4);
  end
  
  s.objects = objects;
  data_set2{c} = s;
  
  c = c + 1;
end

