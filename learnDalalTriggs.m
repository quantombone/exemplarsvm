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
  params2 = esvm_get_default_params;
  f = fields(params);
  for i = 1:length(f)
    params2 = setfield(params2,f{i},getfield(params,f{i}));
  end
  params = params2;
end

params.display = 1;  
params.dump_images = 1;
params.detect_max_windows_per_exemplar = 100;
params.train_max_negatives_in_cache = 5000;
params.train_max_mined_images = 500;
params.latent_iterations = 4;
% for dalaltriggs, it seams having same constant on positives as
% negatives is better than using 50
params.train_positives_constant = 1;
params.mine_from_negatives = 0;
params.mine_from_positives = 1;
params.mine_skip_positive_objects_os = .2;
params.train_max_scale = 1.0;
params.latent_os_thresh = 0.7;

%params.detect_pyramid_padding = 0;

% if ~exist('localdir','var')
%localdir = '/nfs/baikal/tmalisie/ldt/';
% end

% params.localdir = ''; 

%% Issue warning if lock files are present
% lockfiles = check_for_lock_files(localdir);
% if length(lockfiles) > 0
%   fprintf(1,'WARNING: %d lockfiles present in current directory\n', ...
%           length(lockfiles));
% end

% KILL_LOCKS = 1;
% for i = 1:length(lockfiles)
%   unix(sprintf('rmdir %s',lockfiles{i}));
% end


model = esvm_initialize_dt(data_set, cls, params);
model = esvm_train(model);

for niter = 1:params.latent_iterations
  model = esvm_latent_update_dt(model);
  model = esvm_train(model);
end

%% Perform Platt calibration and M-matrix estimation
%M = esvm_perform_calibration(val_grid, val_set, models,val_params);

%% Apply trained exemplars on validation set
%val_grid = esvm_detect_imageset(val_set, models, val_params,
%val_set_name);
                       
% %% Define test-set
% test_params = params;
% test_params.detect_exemplar_nms_os_threshold = 1.0;
% test_params.detect_max_windows_per_exemplar = 500;
% test_params.detect_keep_threshold = -2.0;

% %test_set = load(sprintf('%s/%s/test.mat',...
% %             data_directory, dataset_directory),'data_set');
% %test_set = test_set.data_set;

% test_set = get_objects_set(test_set, cls);
% test_set_name = ['test+' cls];

% %% Apply on test set
% test_grid = esvm_detect_imageset(test_set, models, test_params, test_set_name);

% %% Apply calibration matrix to test-set results
% test_struct = esvm_pool_exemplar_dets(test_grid, models, [], ...
%                                       test_params);

% %% Perform the exemplar evaluation
% results = esvm_evaluate_pascal_voc(test_struct, test_set, models, params, ...
%                                      test_set_name);

% if params.display
%   for mind = 1:length(models)
%     I = esvm_show_top_exemplar_dets(test_struct, test_set, ...
%                                     models, mind,10,10);
%     figure(45)
%     imagesc(I)
%     title('Top detections','FontSize',18);
%     drawnow
%     snapnow
    
%     if params.dump_images == 1 && length(params.localdir)>0
%       filer = sprintf('%s/results/topdet.%s-%04d-%s.png',...
%                       localdir, models{mind}.models_name, mind, test_set_name);
      
%       imwrite(I,filer);
%     end 
%   end
% end


% return;

% %NOTE: the show_top_dets functions doesn't even work for dalaltriggs (but we can apply NN
% %hack! if we want to)


% %% Show top 20 detections as exemplar-inpainting results
% maxk = 20;
% allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
%                             params,  maxk, test_set_name);

