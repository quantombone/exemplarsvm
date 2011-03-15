function [grid,truncated] = prune_occluders(models, grid, truncated)
%Prune highly occluding objects

VOCinit;
if ~exist('truncated','var')
  stuff = cell(length(models),1);
  for i = 1:length(models)
    fprintf(1,'.');
    curid = models{i}.curid;
    recs = PASreadrecord(sprintf(VOCopts.annopath,curid));  
    objectid = models{i}.objectid;
    
    stuff{i} = recs.objects(objectid);
  end
  truncated = cellfun(@(x)x.truncated,stuff);
end

goods = find(truncated==0);

for i = 1:length(grid)
  curhits = ismember(grid{i}.bboxes(:,6), goods);
  grid{i}.coarse_boxes = grid{i}.coarse_boxes(curhits, :);
  grid{i}.bboxes = grid{i}.bboxes(curhits, :);
  grid{i}.extras.os = grid{i}.extras.os(curhits,:);
end
