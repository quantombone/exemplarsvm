classes = {'train','tvmonitor','bottle','cow'};

for i = 1:length(classes)
%% Here is the main script for doing the NN baseline

%Set default class and detector mode, by writing into the default
%file, which our mapreduces will be reading.
save_default_class(classes{i},'exemplars-dt');

%Run an "exemplar_initialize" mapreduce
timing.initialize = spawn_job('ei',50,2);

%Load the models -- which will force a caching of result file
models = load_all_models;

%Run an "apply_voc_exemplars"
timing.apply = spawn_job('ave',120,2);

grid = load_result_grid(models);
clear M
M.betas = perform_calibration(models,grid);
results = evaluate_pascal_voc_grid(models, grid, 'test', M);
results = evaluate_pascal_voc_grid(models, grid, 'test');

end