% Demo: Training Exemplar-SVMs from synthetic data
% This function can generate a nice HTML page by calling: publish('esvm_demo_train_synthetic.m','html')
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

function [models,M] = esvm_demo_train_synthetic

addpath(genpath(pwd))

%% Create a synthetic dataset of circles on a random background
Npos = 20;
Nneg = 50;
[pos_set,neg_set] = esvm_generate_dataset(Npos,Nneg);

%% Set exemplar-initialization parameters
params = esvm_get_default_params;
params.init_params.sbin = 4;
params.init_params.MAXDIM = 6;
params.model_type = 'exemplar';
params.dataset_params.display = 1;
%if localdir is commented out, no local saving happens
%params.dataset_params.localdir = '/nfs/baikal/tmalisie/synthetic/';

%%Initialize exemplar stream
stream_params.stream_set_name = 'trainval';
stream_params.stream_max_ex = 1;
stream_params.must_have_seg = 0;
stream_params.must_have_seg_string = '';
stream_params.model_type = 'exemplar'; %must be scene or exemplar
%assign pos_set as variable
stream_params.pos_set = pos_set;

%% Get the positive stream
e_stream_set = esvm_get_pascal_stream(stream_params, ...
                                      params.dataset_params);

val_neg_set = neg_set((Nneg/2+1):end);
neg_set = neg_set(1:((Nneg/2)));

%% Initialize Exemplars
initial_models = esvm_initialize_exemplars(e_stream_set, params, ...
                                           'initial');

%% Set exemplar-svm training parameters
train_params = params;
train_params.detect_max_scale = 1.0;
train_params.train_max_mined_images = 50;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;

train_params.CACHE_FILE = 1;
%% Perform Exemplar-SVM training
[models] = esvm_train_exemplars(initial_models, ...
                                neg_set, train_params);

val_params = params;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.gt_function = @get_pascal_anno_function;
val_set = cat(1,pos_set(:),val_neg_set(:));
val_set_name = 'valset';

%% Apply trained exemplars on validation set
val_grid = esvm_detect_imageset(val_set, models, val_params, val_set_name);


val_params.val_set = val_set;
%% Perform Platt calibration and M-matrix estimation
M = esvm_perform_calibration(val_grid, models, val_params);


%% Define test-set
test_params = params;
test_params.detect_exemplar_nms_os_threshold = 0.5;
Ntest = 200;
[stream_test] = esvm_generate_dataset(Ntest);
test_set = cellfun2(@(x)x.I,stream_test);

%% Apply on test set
test_grid = esvm_detect_imageset(test_set, models, test_params);

%% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, M, test_params);

%% Show top detections
maxk = 20;
allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
                       params,  maxk);



