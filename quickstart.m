%classes =...
%{'car','dog','cat','bird','chair','pottedplant'};%'tvmonitor','bottle','cow'};
classes = { 'train' };

for i = 1:length(classes)
%% Here is the main script for doing the NN baseline

%Set default class and detector mode, by writing into the default
%file, which our mao,2n-.0mpreduces will be reading.
save_default_class(classes{i},'exemplars-svm');
return;
%Run an "exemplar_initialize" mapreduce
timing.initialize = spawn_job('ei',50,2);

%Load the models -- which will force a caching of result file
models = load_all_models;

%Run an "apply_voc_exemplars"
timing.apply = spawn_job('ave',80,2);

grid = load_result_grid(models);
clear M
M.betas = perform_calibration(models,grid);
[results,final] = evaluate_pascal_voc_grid(models, grid, 'test', M);
%[results,final] = evaluate_pascal_voc_grid(models, grid, 'test');

show_top_transfers(models,grid,'test',final)
end