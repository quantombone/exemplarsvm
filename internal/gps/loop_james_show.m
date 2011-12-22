for i = 1:78
  fprintf(1,'on image %d\n',i);
  models = load_all_exemplars(sprintf('james.%05d',i));
  grid = load_result_grid(models);  
  %[tmp,james_bbs{i}] = perform_calibration(grid,models);
  perform_calibration(grid,models);
end
