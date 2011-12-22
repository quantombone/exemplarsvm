function [models,M,val_grid,test_grid] = ...
    voc_template(dataset_params, e_stream_set, neg_set, ...
                      val_set, test_set, cls)
%% This is the main Exemplar-SVM PASCAL VOC pipeline script, which
%is called from voc_demo_esvm after the parameters of the
%experiment have been set-up

models_name = dataset_params.models_name;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR INITIALIZATION %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

efiles = esvm_initialize(dataset_params, e_stream_set, ...
                         models_name, dataset_params.init_params);

%Append the nn-type if we are in nn mode
if length(dataset_params.params.nnmode) > 0
  models_name = [models_name '-' dataset_params.params.nnmode];
end

%Load all of the initialized exemplars
CACHE_FILE = 1;
STRIP_FILE = 0;
models = esvm_load_models(dataset_params, cls, models_name, ...
                         efiles, CACHE_FILE, STRIP_FILE);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR TRAINING %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Train each initialized exemplar     
[tfiles, models_name] = esvm_train(dataset_params, ...
                                   models, neg_set);  

if isfield(dataset_params, 'JUST_TRAIN') && ...
      (dataset_params.JUST_TRAIN==1)
  fprintf(1,'only training because JUST_TRAIN is enabled\n');
  return;
end

%Load the trained exemplars (this will hold script until all
%exemplars have been trained)
CACHE_FILE = 1;
STRIP_FILE = 1;
models = esvm_load_models(dataset_params, cls, models_name, ...
                          tfiles, CACHE_FILE, STRIP_FILE);

if isfield(dataset_params, 'JUST_TRAIN_AND_LOAD') && ...
      (dataset_params.JUST_TRAIN_AND_LOAD ==1 )
  fprintf(1,'only train+load because JUST_TRAIN_AND_LOAD is enabled\n');
  return;
end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR CROSS VALIDATION %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply trained exemplars on validation set
dataset_params.params = dataset_params.val_params;
dataset_params.params.gt_function = @get_pascal_anno_function;
val_files = esvm_detect_set(dataset_params, models, val_set, ...
                            dataset_params.val_params.set_name);

if isfield(dataset_params, 'JUST_APPLY') && ...
      (dataset_params.JUST_APPLY==1)
  fprintf(1,'only applying because JUST_APPLY is enabled\n');
  %do nothing
else
  
  %Load validation results
  val_grid = esvm_load_result_grid(dataset_params, models, ...
                                   dataset_params.val_params.set_name, val_files);
    
  %val_struct is not used
  %val_struct = pool_exemplar_detections(dataset_params, models, val_grid);
  
  %% Perform l.a.b.o.o. calibration and M-matrix estimation
  CACHE_BETAS = 1;
  M = esvm_calibrate_with_matrix(dataset_params, models, ...
                                 val_grid, val_set, CACHE_BETAS);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR TESTING %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply trained exemplars on test set
if length(test_set) == 0
  fprintf(1,'Warning, testset is empty\n');
  return;
end

%Apply on test set
dataset_params.params = dataset_params.test_params;
dataset_params.params.gt_function = [];
test_files = esvm_detect_set(dataset_params, models, test_set, ...
                             dataset_params.test_params.set_name);

if isfield(dataset_params, 'JUST_APPLY') && ...
      (dataset_params.JUST_APPLY==1)
  fprintf(1,'only applying because JUST_APPLY is enabled\n');
  return;
end

%Load test results
test_grid = esvm_load_result_grid(dataset_params, models, ...
                                  dataset_params.test_params.set_name, ...
                                  test_files);

test_struct = pool_exemplar_detections(dataset_params, models, test_grid, M);

%[results] = evaluate_pascal_voc_grid(dataset_params, ...
%                                     models, test_grid, ...
%                                     dataset_params.test_params.set_name, ...
%                                     test_struct);
%rc = results.corr;
%test_struct.rc = rc;

maxk = 10;
allbbs = show_top_dets(dataset_params, models, test_grid, test_set, dataset_params.test_params.set_name, ...
                         test_struct, maxk);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR EVALUATION/DISPLAY %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Show all raw detections on test-set as a "memex browser"
%show_exemplar_browser(dataset_params, models, ...
%                      val_grid, val_set, ...
%                      test_grid, test_set, M);
%fprintf(1,'quitting after memex display\n');
%return;
%If no calibration was performed, then we dont do calibrated rounds
if length(M) > 0
  
  if 1
  % %% Evaluation of laboo + M matrix
  test_struct = pool_exemplar_detections(dataset_params, models, test_grid, M);
  
  rc = [];
  if (dataset_params.SKIP_EVAL == 0)
    
    [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                         models, test_grid, ...
                                         dataset_params.test_params.set_name, ...
                                         test_struct);
    rc = results.corr;
  end
  end
end

