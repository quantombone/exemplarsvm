
%%% VOC README
%1. dump out model files for a VOC category
initialize_voc_exemplars('cow');


%2. train all exemplars (all exemplars in exemplar directory)
train_all_exemplars;

return

%3. perform calibration
models = load_all_exemplars;
grid = load_result_grid(models);
betas = perform_calibration(grid,models);

%4. evaluate results on PASCAL VOC comp3 task
results = evaluate_pascal_voc_grid(models, grid, 'test', betas);

%5. show top detections on testset
show_top_dets(models,grid,'test',betas);
