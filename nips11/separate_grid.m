function separate_grid(models,fullgrid)
%% separate the current grid


ac = cellfun2(@(x)x.coarse_boxes,fullgrid);
ac = cat(1,ac{:});
for i = 1:length(models)
  coarse_boxes = ac(ac(:,6)==i,:);
  % tic
  % for j = 1:length(grid)
  %   g = (grid{j}.bboxes(:,6)==i);
  %   grid{j}.coarse_boxes = grid{j}.coarse_boxes(g,:);
  %   grid{j}.bboxes = grid{j}.bboxes(g,:);
  %   grid{j} = rmfield(grid{j},'extras');
  % end
  % toc
 
  filer = sprintf('/nfs/baikal/tmalisie/grids/%05d.mat',i);
  save(filer,'coarse_boxes');
  fprintf(1,'.');
end

