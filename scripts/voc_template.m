function voc_template(dataset_params, cls)
%% This is the main VOC driver script for both scenes and exemplars


if ~exist(dataset_params.devkitroot,'dir')
  mkdir(dataset_params.devkitroot);
end
%Save the parameters so we know later how we generated this run
save([dataset_params.devkitroot '/dataset_params.mat'], ...
     'dataset_params')

models_name = dataset_params.models_name;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR INITIALIZATION %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize exemplars with the exemplar stream
e_stream_set = get_pascal_stream(dataset_params, cls);
efiles = exemplar_initialize(dataset_params, e_stream_set, ...
                             models_name, dataset_params.init_params);

%Append the nn-type if we are in nn mode
if length(dataset_params.params.nnmode) > 0
  models_name = [models_name '-' dataset_params.params.nnmode];
end

%Load all of the initialized exemplars
CACHE_FILE = 1;
STRIP_FILE = 0;
models = load_all_models(dataset_params, cls, models_name, ...
                         efiles, CACHE_FILE, STRIP_FILE);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR TRAINING %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Train each initialized exemplar 
if isfield(dataset_params,'mining_params')
  curparams = dataset_params.mining_params;

  train_set = get_pascal_set(dataset_params, ...
                           curparams.set_name);
  
  if isfield(curparams,'set_maxk')
    train_set = train_set(1:min(length(train_set), ...
                                curparams.set_maxk));
  end
    
  [tfiles, models_name] = train_all_exemplars(dataset_params, ...
                                              models, train_set);  
  
  if isfield(dataset_params, 'JUST_TRAIN') && ...
        (dataset_params.JUST_TRAIN==1)
    fprintf(1,'only training because JUST_TRAIN is enabled\n');
    return;
  end
  %Load the trained exemplars (this will hold script until all
  %exemplars have been trained)
  CACHE_FILE = 1;
  STRIP_FILE = 1;
  models = load_all_models(dataset_params, cls, models_name, ...
                           tfiles, CACHE_FILE, STRIP_FILE);
  
  if isfield(dataset_params, 'JUST_TRAIN_AND_LOAD') && ...
        (dataset_params.JUST_TRAIN_AND_LOAD ==1 )
    fprintf(1,'only train+load because JUST_TRAIN_AND_LOAD is enabled\n');
    return;
  end
  
else
  fprintf(1,['Skipping training because dataset_params.mining_params not' ...
             ' present\n']);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR CROSS VALIDATION %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply trained exemplars on validation set
if isfield(dataset_params,'val_params')
  curparams = dataset_params.val_params;
  val_set = get_pascal_set(dataset_params, ...
                           curparams.set_name);
  if isfield(curparams,'set_maxk')
    val_set = val_set(1:min(length(val_set), ...
                            curparams.set_maxk));
  end

  dataset_params.params = curparams;
  dataset_params.params.gt_function = @get_pascal_anno_function;
  val_files = apply_all_exemplars(dataset_params, models, val_set, ...
                                  curparams.set_name);

  if isfield(dataset_params, 'JUST_APPLY') && ...
        (dataset_params.JUST_APPLY==1)
    fprintf(1,'only applying because JUST_APPLY is enabled\n');
    %do nothing
  else
  
    %Load validation results
    val_grid = load_result_grid(dataset_params, models, ...
                                curparams.set_name, val_files);
    
    %val_struct is not used
    %val_struct = pool_exemplar_detections(dataset_params, models, val_grid);
    
    %% Perform l.a.b.o.o. calibration and M-matrix estimation
    CACHE_BETAS = 1;
    M = calibrate_and_estimate_M(dataset_params, models, ...
                                 val_grid, val_set, CACHE_BETAS);
    
  end
else
  fprintf(1,['Skipping validation becuase dataset_params.val_params not' ...
             ' present\n']);
  M = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR TESTING %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply trained exemplars on test set
if isfield(dataset_params,'test_params')
  curparams = dataset_params.test_params;
  test_set = get_pascal_set(dataset_params, ...
                           curparams.set_name);

  if isfield(curparams,'set_maxk')
    test_set = test_set(1:min(length(test_set), ...
                            curparams.set_maxk));
  end
  
  if length(test_set) == 0
    fprintf(1,'Warning, testset is empty\n');
    return;
  end

  %Apply on test set
  dataset_params.params = curparams;
  dataset_params.params.gt_function = [];
  test_files = apply_all_exemplars(dataset_params, models, test_set, ...
                                  curparams.set_name);
  
  if isfield(dataset_params, 'JUST_APPLY') && ...
        (dataset_params.JUST_APPLY==1)
    fprintf(1,'only applying because JUST_APPLY is enabled\n');
    return;
  end

  %Load test results
  test_grid = load_result_grid(dataset_params, models, ...
                               curparams.set_name, test_files);
  
  %Show all raw detections on test-set as a "memex browser"
  %show_memex_browser2(dataset_params, models, test_grid,...
  %                   test_set, curparams.set_name);


else
  fprintf(1,['Skipping testing becuase dataset_params.test_params not' ...
             ' present\n']);
  
  %If testing is not performed, there is nothing left to do
  return;
end

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
  
  % %% Evaluation of laboo + M matrix
  test_struct = pool_exemplar_detections(dataset_params, models, test_grid, M);
  
  rc = [];
  if (dataset_params.SKIP_EVAL == 0)
    
    [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                         models, test_grid, ...
                                         curparams.set_name, ...
                                         test_struct);
    rc = results.corr;
  end
  
  show_memex_browser2(dataset_params, models, test_struct,...
                      test_set, curparams.set_name, rc);

  
  %% Show top detections for laboo + M matrix
  %show_top_dets(dataset_params, models, test_grid,...
  %              test_set, curparams.set_name, ...
  %              test_struct);
  
  %% Evaluation of l.a.b.o.o. afer training
  M2 = [];
  M2.betas = M.betas;
  test_struct = pool_exemplar_detections(dataset_params, models, test_grid, ...
                             M2);
  
  rc = [];
  if (dataset_params.SKIP_EVAL == 0)
    [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                         models, test_grid, ...
                                         curparams.set_name,...
                                         test_struct);
    
    %% Show top detections from l.a.b.o.o.
    %show_top_dets(dataset_params, models, test_grid,...
    %              test_set, dataset_params.testset_name, ...
    %              test_struct);
    rc = results.corr;
  end  

  show_memex_browser2(dataset_params, models, test_struct,...
                      test_set, curparams.set_name, rc);

end

%% Evaluation of uncalibrated SVM classifiers
M2 = [];
test_struct = pool_exemplar_detections(dataset_params, models, test_grid, M2);

rc = [];
if (dataset_params.SKIP_EVAL == 0)
  [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                       models, test_grid, ...
                                       curparams.set_name, ...
                                       test_struct);
  rc = results.corr;
end

show_memex_browser2(dataset_params, models, test_struct,...
                    test_set, curparams.set_name, rc);

%%% Show top detections from uncalibrated SVM classifiers
% show_top_dets(dataset_params, models, test_grid,...
%               test_set, dataset_params.testset_name, ...
%               test_struct);
