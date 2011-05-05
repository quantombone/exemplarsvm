%%% training a cow detector

%1. dump out model files for a VOC category
%initialize_voc_exemplars('cow');

%Set default class, by writing into the default file, which our
%mapreduces will be reading.
save_default_class('bus');

%Run an "exemplar_initialize" mapreduce
timing.initialize = spawn_job('ei',50,2);

%Load the models -- which will force a caching of result file
models = load_all_models;

%Run an "apply_voc_exemplars"
timing.apply = spawn_job('ave',80,2);

models = load_all_models;
grid = load_result_grid(models);
betas = perform_calibration(grid,models);
results = evaluate_pascal_voc_grid(models, grid, 'test');

%2. train all exemplars (all exemplars in exemplar directory)
% make sure mining_params has dump_images enabled if you want to
% look at the exemplars
%mining_params.dump_images = 1;
%train_all_exemplars;

return

%3. perform calibration
models = load_all_exemplars;
grid = load_result_grid(models);
betas = perform_calibration(grid,models);

%4. evaluate results on PASCAL VOC comp3 task
results = evaluate_pascal_voc_grid(models, grid, 'test', betas);

%5. show top detections on testset
show_top_dets(models,grid,'test',betas);
