function [model] = learnExemplarSVMs(data_set, cls, params)
% Learn an Ensemble of Exemplar-SVMs for a single class in the data set
% Performs evaluation on test_set [optional].
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

addpath(genpath(pwd));

if ~exist('cls','var')
  error('Needs class as input');
end

data_directory = '/Users/tomasz/projects/pascal/';
dataset_directory = 'VOC2007';

if ~exist('data_set','var') || length(data_set) == 0
  load(sprintf('%s/%s/trainval.mat',...
               data_directory, dataset_directory),'data_set');
end

if ~exist('params','var') || length(params) == 0
  %% Get default parameters
  params = esvm_get_default_params;
  params.display = 1;  
  params.dump_images = 0;
  params.detect_max_scale = 0.5;
  params.detect_exemplar_nms_os_threshold = 1.0; 
  params.detect_max_windows_per_exemplar = 100;
  params.train_max_negatives_in_cache = 5000;
  params.train_max_mined_images = 50;
  params.detect_pyramid_padding = 0;
end

%should be esvm_initialize_exemplarSVMs
model = esvm_initialize_exemplars(data_set, cls, params);

%perform training
model = esvm_train(model);

%% Apply trained exemplars on validation set
val_grid = esvm_detect_imageset(data_set, model, params);%, val_set_name);
                       
%% Perform Platt calibration and M-matrix estimation
model.M = esvm_perform_calibration(val_grid, data_set, model, ...
                                   params);


