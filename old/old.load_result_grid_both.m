function grid_both = load_result_grid_both(models,curset)
grid2 = load_result_grid(models,['FLIP-' curset]);
grid1 = load_result_grid(models,curset);

grid_both = grid1;
for i = 1:length(grid1)
  coarse_boxes = grid2{i}.coarse_boxes;
  bboxes = grid2{i}.bboxes;
  
  coarse_boxes(:,6) = abs(coarse_boxes(:,6))+length(models)/2;
  bboxes(:,6) = abs(bboxes(:,6))+length(models)/2;
  
  grid_both{i}.coarse_boxes = cat(1,grid_both{i}.coarse_boxes,...
                                  coarse_boxes);
  
  grid_both{i}.bboxes = cat(1,grid_both{i}.bboxes,...
                                  bboxes);
  
  if isfield(grid2{i},'extras') && isfield(grid2{i}.extras,'os')
    os = grid2{i}.extras.os;
    grid_both{i}.extras.os = cat(1,grid_both{i}.extras.os, ...
                                 os);
  end
  
end
