% Demo: Training Exemplar-SVMs from synthetic data
% This function can generate a nice HTML page by calling: publish('esvm_demo_train_synthetic.m','html')
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

function [models,M] = esvm_demo_train_synthetic

%% Create a synthetic dataset of circles on a random background
Npos = 1;
Nneg = 50;
Ntest = 50;
[e_stream_set,neg_set] = esvm_generate_dataset(Npos,Nneg);

%% Set exemplar-initialization parameters
params = esvm_get_default_params;
params.init_params.sbin = 4;
params.model_type = 'exemplar';
params.dataset_params.display = 1;

%% Initialize Exemplars
initial_models = esvm_initialize_exemplars(e_stream_set, params);

%% Set exemplar-svm training parameters
train_params = params;
train_params.detect_max_scale = 1.0;
train_params.train_max_mined_images = 50;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;

%% Perform Exemplar-SVM training
[models] = esvm_train_exemplars(initial_models, ...
                                neg_set, train_params);

% val_params = params;
% val_params.detect_exemplar_nms_os_threshold = 0.5;
% val_params.gt_function = @get_pascal_anno_function;
% val_params.CACHE_BETAS = 1;
% val_set = get_pascal_set(dataset_params, val_set_name);
% val_set = val_set(1:10);

%% Apply trained exemplars on validation set
%val_grid = esvm_detect_imageset(val_set, models, val_params, val_set_name);

%% Perform Platt calibration and M-matrix estimation
%M = esvm_perform_calibration(val_grid, models, val_params);

%% Define test-set
test_params = params;
test_params.detect_exemplar_nms_os_threshold = 0.5;
[stream_test] = esvm_generate_dataset(Ntest);
test_set = cellfun2(@(x)x.I,stream_test);

%% Apply on test set
test_grid = esvm_detect_imageset(test_set, models, test_params);

%% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, [], test_params);

%% Show top detections
maxk = 20;
allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
                       params,  maxk);



