function val_grid = prune_grid(val_grid, exemplar_subset)
%Take all of the detections inside a grid of detections "val_grid",
%and prune the detections to only keep detections from the exemplar
%ids inside exemplar_subset.  This is used to load all exemplar
%detections, then choose a subset for thesis evaluation.
%Tomasz Malisiewicz (tomasz@cmu.edu)

for zzz = 1:length(val_grid)
  goods = find(ismember(val_grid{zzz}.bboxes(:,6), ...
                        exemplar_subset));
  val_grid{zzz}.bboxes = val_grid{zzz}.bboxes(goods,:);
  val_grid{zzz}.coarse_boxes = ...
      val_grid{zzz}.coarse_boxes(goods,:);
  
  [aa,bb] = ismember(val_grid{zzz}.bboxes(:,6),exemplar_subset);
  val_grid{zzz}.bboxes(:,6) = bb;
  val_grid{zzz}.coarse_boxes(:,6) = bb;
  
  try
    val_grid{zzz}.extras.maxos = ...
        val_grid{zzz}.extras.maxos(goods);
    val_grid{zzz}.extras.maxind = ...
        val_grid{zzz}.extras.maxind(goods);
    val_grid{zzz}.extras.maxclass = ...
        val_grid{zzz}.extras.maxclass(goods);
  catch
  end
end
