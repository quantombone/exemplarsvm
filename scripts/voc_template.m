function voc_template(dataset_params, cls)
%% This is the main VOC driver script for both scenes and exemplars

e_stream_set = get_pascal_stream(dataset_params, cls);

train_set = get_pascal_set(dataset_params, ...
                           dataset_params.trainset_name,...
                           dataset_params.trainset_name2);
train_set = train_set(1:min(length(train_set), dataset_params.trainset_maxk));

val_set = get_pascal_set(dataset_params, ...
                         dataset_params.valset_name,...
                         dataset_params.valset_name2);
val_set = val_set(1:min(length(val_set), dataset_params.valset_maxk));

if strcmp(dataset_params.model_type,'exemplar')
  val_gt_function = @get_pascal_anno_function;
else
  val_gt_function = [];
end
val_params = get_default_mining_params;

test_set = get_pascal_set(dataset_params, ...
                          dataset_params.testset_name,...
                          dataset_params.testset_name2);
test_set = test_set(1:min(length(test_set), dataset_params.testset_maxk));

test_gt_function = [];
test_params = get_default_mining_params;

if length(test_set) == 0
  fprintf(1,'Warning, testset is empty\n');
  return;
end

models_name = dataset_params.models_name;
init_params = dataset_params.init_params;

%% Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(dataset_params, e_stream_set, ...
                             models_name, init_params);

%Load all of the initialized exemplars
models = load_all_models(dataset_params, cls, models_name, efiles, 1);

%% Train each initialized exemplar
[tfiles, models_name] = train_all_exemplars(dataset_params, models, ...
                                            train_set, ...
                                            dataset_params.mining_params, ...
                                            dataset_params.training_function);

%Load the trained exemplars
models = load_all_models(dataset_params, cls, models_name, tfiles, ...
                         1, 1);

%% Apply trained exemplars on test set
test_files = apply_all_exemplars(dataset_params, models, test_set, ...
                                 dataset_params.testset_name, ...
                                 [], test_params, ...
                                 test_gt_function);


%Load testset results
test_grid = load_result_grid(dataset_params, models, ...
                             dataset_params.testset_name, test_files);

%% Evaluation uncalibrated SVM classifiers
teststruct = pool_results(dataset_params, models, test_grid);
if strcmp(dataset_params.model_type,'exemplar') && ...
      (dataset_params.SKIP_EVAL == 0)
  [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                       models, test_grid, ...
                                       dataset_params.testset_name, teststruct);
end

show_memex_browser(dataset_params, models, test_grid,...
                   test_set, dataset_params.testset_name, ...
                   teststruct);

%%% Show top detections from uncalibrated SVM classifiers
% show_top_dets(dataset_params, models, test_grid,...
%               test_set, dataset_params.testset_name, ...
%               teststruct);

if dataset_params.SKIP_VALIDATION == 1
  return;
end


if strcmp(dataset_params.model_type,'exemplar')
  %% Apply trained exemplars on validation set
  val_files = apply_all_exemplars(dataset_params, models, val_set, ...
                                  dataset_params.valset_name, ...
                                  [], val_params, val_gt_function);

  %Load validation results
  val_grid = load_result_grid(dataset_params, models, ...
                              dataset_params.valset_name, val_files);
  
  %% Perform l.a.b.o.o. calibration and M-matrix estimation
  M = calibrate_and_estimate_M(dataset_params, models, val_grid);
  
else
  M = [];
end

if length(M) > 0 && (dataset_params.SKIP_EVAL == 0)
  %% Evaluation of l.a.b.o.o. afer training
  M2.betas = M.betas;
  teststruct = pool_results(dataset_params, models, test_grid, M2);
  [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                       models, test_grid, ...
                                       dataset_params.testset_name, teststruct);
  
  %% Show top detections from l.a.b.o.o.
  show_top_dets(dataset_params, models, test_grid,...
                test_set, dataset_params.testset_name, ...
                teststruct);
  
  %% Evaluation of laboo + M matrix
  teststruct = pool_results(dataset_params, models, test_grid, M);
  
  if strcmp(dataset_params.model_type,'exemplar') && ...
        (dataset_params.SKIP_EVAL == 0)
    [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                         models, test_grid, ...
                                         dataset_params.testset_name, ...
                                         teststruct);
  end
  
  %% Show top detections for laboo + M matrix
  show_top_dets(dataset_params, models, test_grid,...
                test_set, dataset_params.testset_name, ...
                teststruct);
end
