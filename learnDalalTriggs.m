function model = learnDalalTriggs(data_set, cls, params)
% Learn a DalalTriggs template detector, with latent updates,
% perturbed assignments (which help avoid local minima), and
% ability to use different features.
%
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

if length(data_set) == 0
  fprintf(1,'No dataset provided, loading default DATA');
  
  data_directory = '/Users/tomasz/projects/pascal/';
  dataset_directory = 'VOC2007';  
  data_set = install_dataset(data_directory, dataset_directory);
end

if ~exist('params','var') || length(params) == 0
  %% Get default parameters
  params = esvm_get_default_params;
else
  params = esvm_get_default_params(params);
end

params.display = 1;  
params.dump_images = 0;
params.detect_max_windows_per_exemplar = 200;
params.train_max_negatives_in_cache = 20000;
params.max_number_of_positives = 1000;
params.train_max_mined_images = 10000;
params.latent_iterations = 2;
params.train_svm_c = .1;
params.train_max_windows_per_iteration = 3000;
% for dalaltriggs, it seams having same constant on positives as
% negatives is better than using 50
params.train_positives_constant = 1;
params.mine_from_negatives = 0;
params.mine_from_positives = 1;
params.mine_skip_positive_objects_os = .2;
params.mine_from_positives_do_latent_update = 1;
params.train_max_scale = 1.0;
params.latent_os_thresh = 0.5;
params.dt_initialize_with_flips = 0;

%params.detect_pyramid_padding = 0;
%fprintf(1,'hack max 40\n');
%params.max_number_of_positives = 20;

starttime = tic;
model = esvm_initialize_dt(data_set, cls, params);

params.mine_from_positives = 1;
params.mine_from_negatives = 1;
params.train_max_mined_images = 3000;

model.params = esvm_get_default_params(params);
model = esvm_train(model);

% params.train_max_mined_images = 1000;
% params.mine_from_negatives = 1;
% params.mine_from_positives = 0;
% model.params = esvm_get_default_params(params);
% model = esvm_train(model);

if 0
for niter = 1:params.latent_iterations
  %model = esvm_latent_update_dt(model);
  params.mine_from_positives = 1;
  params.mine_from_negatives = 0;
  model.params = esvm_get_default_params(params);
  model = esvm_train(model);
  
  params.mine_from_positives = 0;
  params.mine_from_negatives = 1;
  model.params = esvm_get_default_params(params);
  model = esvm_train(model);
end
end

model.learn_time = toc(starttime);
