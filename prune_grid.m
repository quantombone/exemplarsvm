function [models,grid] = prune_grid(models,grid,subset)

models = models(subset);
for i = 1:length(grid)
  [oks,newids] = ismember(grid{i}.bboxes(:,6), subset);
  bboxes = grid{i}.bboxes(oks,:);
  if sum(oks) > 0  
    bboxes(:,6) = newids(oks);
  end
  
  grid{i}.bboxes = bboxes;
  %grid{i}
end
