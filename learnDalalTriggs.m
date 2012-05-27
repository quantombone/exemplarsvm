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

if ~exist('data_set','var')
  data_set = load('~/projects/pascal/VOC2007/trainval.mat');
  data_set = data_set.data_set;
end

if ~exist('cls','var')
  cls = 'bus';
end

fprintf(1,'Class is %s\n',cls);
  
if ~exist('params','var')
  params = esvm_get_default_params;
end

params.display = 0;
params.dump_images = 0;
params.detect_max_windows_per_exemplar = 500;
params.max_number_of_positives = 2000;

params.train_max_negatives_in_cache = 30000;
params.train_max_mined_images = length(data_set);
params.train_max_negative_images = 10000;
params.train_max_mine_iterations = 500;

params.train_svm_c = .01;
params.train_max_windows_per_iteration = 3000;
params.train_positives_constant = 1;
params.train_max_scale = 1.0;
params.train_max_images_per_iteration = 200;
params.detect_pyramid_padding = 5;
params.detect_levels_per_octave = 10;

params.mine_from_negatives = 1;
params.mine_from_positives = 1;
params.mine_skip_positive_objects_os = .1;
params.mine_from_positives_do_latent_update = 0;

params.latent_os_thresh = 0.5;
params.init_params.MAX_POS_CACHE = 5000;
params.init_params.ADD_LR = 1;
%params.init_params.init_function = ...
%    @esvm_initialize_fixedframe_exemplar;
params.training_function = @ ...
    (m)esvm_update_positives(esvm_update_svm(m,1,10000),1,1,1,10000);

params.training_function = @ ...
    (m)esvm_update_positives(esvm_update_svm(esvm_update_positives(esvm_update_svm(m,.8,10000),1,1,1,10000),1,10000),1,1,1,10000);

%hardcode mask size
%params.init_params.hg_size = [10 10 31];

model = esvm_initialize_exemplars(data_set, cls, params);
model.params.display = 1;

model = unify_model(model);

params.display = 1;
model.params = params;

model.models{1}.params = params;
model.models{1}.params.train_svm_c = model.params.train_svm_c;
model.models{1}.params.train_newton_iter = 10;
model.params.train_newton_iter = 10;

starter = tic;
model = esvm_train(model);
model.learn_time = toc(starter);
