function voc_template(dataset_params, cls)
%% This is the main VOC driver script for both scenes and exemplars

models_name = dataset_params.models_name;

if ~isfield(dataset_params,'train_params')
  dataset_params.train_params = dataset_params.params;
end

if ~isfield(dataset_params,'val_params')
  dataset_params.val_params = dataset_params.params;
end

if ~isfield(dataset_params,'test_params')
  dataset_params.test_params = dataset_params.params;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR INITIALIZATION %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize exemplars with the exemplar stream
e_stream_set = get_pascal_stream(dataset_params, cls);
efiles = exemplar_initialize(dataset_params, e_stream_set, ...
                             models_name, dataset_params.init_params);

%Load all of the initialized exemplars
CACHE_FILE = 0;
STRIP_FILE = 0;
models = load_all_models(dataset_params, cls, models_name, ...
                         efiles, CACHE_FILE, STRIP_FILE);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR TRAINING %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Train each initialized exemplar 
if (dataset_params.SKIP_TRAINING == 0)
  train_set = get_pascal_set(dataset_params, ...
                             dataset_params.trainset_name,...
                             dataset_params.trainset_name2);
  train_set = train_set(1:min(length(train_set), ...
                              dataset_params.trainset_maxk));

  [tfiles, models_name] = train_all_exemplars(dataset_params, models, ...
                                              train_set);  
  %Load the trained exemplars
  CACHE_FILE = 1;
  STRIP_FILE = 1;
  models = load_all_models(dataset_params, cls, models_name, ...
                           tfiles, CACHE_FILE, STRIP_FILE);
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR CROSS VALIDATION %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply trained exemplars on validation set
if strcmp(dataset_params.model_type,'exemplar') && ...
      (dataset_params.SKIP_VAL == 0)

  val_set = get_pascal_set(dataset_params, ...
                           dataset_params.valset_name,...
                           dataset_params.valset_name2);
  val_set = val_set(1:min(length(val_set), dataset_params.valset_maxk));
  
  %Apply on validation set
  dataset_params.params = dataset_params.val_params;
  dataset_params.params.gt_function = @get_pascal_anno_function;
  val_files = apply_all_exemplars(dataset_params, models, val_set, ...
                                  dataset_params.valset_name, []);

  %Load validation results
  val_grid = load_result_grid(dataset_params, models, ...
                              dataset_params.valset_name, val_files);
  
  %% Perform l.a.b.o.o. calibration and M-matrix estimation
  M = calibrate_and_estimate_M(dataset_params, models, val_grid);
else
  M = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR TESTING %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_set = get_pascal_set(dataset_params, ...
                          dataset_params.testset_name,...
                          dataset_params.testset_name2);
test_set = test_set(1:min(length(test_set), dataset_params.testset_maxk));

if length(test_set) == 0
  fprintf(1,'Warning, testset is empty\n');
  return;
end

%Apply trained exemplars on test set
dataset_params.params = dataset_params.test_params;
dataset_params.params.gt_function = [];
test_files = apply_all_exemplars(dataset_params, models, test_set, ...
                                 dataset_params.testset_name,[]);

%Load testset results
test_grid = load_result_grid(dataset_params, models, ...
                             dataset_params.testset_name, test_files);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR EVALUATION %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Evaluation uncalibrated SVM classifiers
teststruct = pool_results(dataset_params, models, test_grid);
if strcmp(dataset_params.model_type,'exemplar') && ...
      (dataset_params.SKIP_EVAL == 0)
  [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                       models, test_grid, ...
                                       dataset_params.testset_name, teststruct);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EVALUATION DISPLAY %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

show_memex_browser(dataset_params, models, test_grid,...
                   test_set, dataset_params.testset_name);

%%% Show top detections from uncalibrated SVM classifiers
% show_top_dets(dataset_params, models, test_grid,...
%               test_set, dataset_params.testset_name, ...
%               teststruct);

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
