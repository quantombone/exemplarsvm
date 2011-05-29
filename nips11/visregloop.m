%classes =...
%{'car','dog','cat','bird','chair','pottedplant'};%'tvmonitor','bottle','cow'};
%classes = { 'motorbike' };

cls = 'motorbike';
%cls = 'train';
mode = 'exemplars';

%% Here is the main script for doing the NN baseline

%Set default class and detector mode, by writing into the default
%file, which our mapreduces will be reading.
save_default_class(cls,'exemplars');

%Run an "exemplar_initialize" mapreduce
%timing.initialize = spawn_job('ei',50,2);

timing.initialize = spawn_job('ei',50,2);

timing.rank1 = spawn_job('rank_all_exemplars',50,2);

save_default_class(cls,'exemplars-rank2');

%Create initialization of file, as well as stripped file, by
%loading without output arguments
load_all_models;

%apply everything on trainval
timing.app1 = spawn_job('ave_trainval',50,2);
models = load_all_models;
grid = load_result_grid_trainval(models);

separate_grid(models,grid);

%Run the visual sim learning task
timing.app1 = spawn_job('reregt',50,4);


return;

for q = 1:10
  %Load the models -- which will force a caching of result file
  models = load_all_models;
  
  timing.apply = spawn_job('ave2',80,2);
  res = load_result_grid(models);
  
  timing.train = spawn_job('do_train_mr',40,4);
  mode = [mode 'I'];
  save_default_class(cls,mode);
end


return
%Run an "apply_voc_exemplars"
timing.apply = spawn_job('ave2',80,2);

grid = load_result_grid(models);
clear M
M.betas = perform_calibration(models,grid);
[results,final] = evaluate_pascal_voc_grid(models, grid, 'test', M);
%[results,final] = evaluate_pascal_voc_grid(models, grid, 'test');

show_top_transfers(models,grid,'test',final)
