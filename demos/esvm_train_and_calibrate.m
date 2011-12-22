function [models, M, val_grid] = ...
    esvm_train_and_calibrate(dataset_params, e_stream_set, neg_set, ...
                      val_set, cls)
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
%Load the trained exemplars (this will hold script until all
%exemplars have been trained)
CACHE_FILE = 1;
STRIP_FILE = 1;
models = esvm_load_models(dataset_params, cls, models_name, ...
                          tfiles, CACHE_FILE, STRIP_FILE);

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% EXEMPLAR CALIBRATION %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Apply trained exemplars on validation set
dataset_params.params = dataset_params.val_params;
dataset_params.params.gt_function = @get_pascal_anno_function;
val_files = esvm_detect_set(dataset_params, models, val_set, ...
                            dataset_params.val_params.set_name);
val_grid = esvm_load_result_grid(dataset_params, models, ...
                                 dataset_params.val_params.set_name, val_files);

%% Perform l.a.b.o.o. calibration and M-matrix estimation
CACHE_BETAS = 1;
M = esvm_calibrate_with_matrix(dataset_params, models, ...
                               val_grid, val_set, CACHE_BETAS);

