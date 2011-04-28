function loop_sketches_show;
VOCinit;
classes = VOCopts.classes;

classes = {'car'};

for i = 1:length(classes)
  models = load_all_exemplars(classes{i});
  if length(models) == 0
    continue
  end
      
  grid = load_result_grid(models);
  perform_calibration(grid,models);
end
