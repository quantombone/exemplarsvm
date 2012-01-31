function [models,M] = esvm_script_train_dt(cls, ...
                                           data_directory, ...
                                           dataset_directory, ...
                                           results_directory)

% Script: PASCAL VOC training/testing script
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

addpath(genpath(pwd));

if ~exist('cls','var')
  cls = 'bus';
end

if ~exist('data_directory','var')
  data_directory = '/Users/tomasz/projects/pascal/';
  data_directory = '/csail/vision-videolabelme/databases/';
  data_directory = '/csail/vision-videolabelme/people/tomasz/VOCdevkit/';
end

if ~exist('dataset_directory','var')
  dataset_directory = 'VOC2007';
end

if ~exist('results_directory','var')

  results_directory = sprintf(['/csail/vision-videolabelme/people/tomasz/newdt/dt-%s-' ...
                    '%s/'], ...
                              dataset_directory, cls);
end

%% Initialize dataset parameters
%data_directory = '/Users/tomasz/projects/Pascal_VOC/';
%results_directory = '/nfs/baikal/tmalisie/esvm-data/';

%data_directory = '/csail/vision-videolabelme/people/tomasz/VOCdevkit/';
%results_directory = sprintf('/csail/vision-videolabelme/people/tomasz/esvm-%s/',cls);

dataset_params = esvm_get_voc_dataset(dataset_directory, ...
                                      data_directory, ...
                                      results_directory);
dataset_params.display = 1;
%dataset_params.dump_images = 1;

%% Issue warning if lock files are present
lockfiles = check_for_lock_files(results_directory);
if length(lockfiles) > 0
  fprintf(1,'WARNING: %d lockfiles present in current directory\n', ...
          length(lockfiles));
end

% KILL_LOCKS = 1;
% for i = 1:length(lockfiles)
%   unix(sprintf('rmdir %s',lockfiles{i}));
% end

%% Get default parameters
params = esvm_get_default_params;
params.dataset_params = dataset_params;

% for dalaltriggs, it seams having same constant on positives is better
params.train_positives_constant = 1;


%% Get positive set
pos_set = esvm_get_pascal_set(dataset_params, ['trainval+' cls]);
pos_annos = emap(@(x)PASreadrecord(strrep(strrep(x,'JPEGImages','Annotations'),'.jpg','.xml')),pos_set);
pos_set = cellfun(@(x,y)setfield(x,'I',y),pos_annos,pos_set, ...
                  'UniformOutput',false);
goods = cellfun(@(x)find((ismember({x.objects.class},cls) & ([x.objects.difficult]==0))),pos_set, ...
                'UniformOutput',false);
pos_set = cellfun(@(x,y)setfield(x,'objects',x.objects(y)),pos_set,goods,'UniformOutput',false);

%% Get negative set
neg_set = esvm_get_pascal_set(dataset_params, ['trainval-' cls]);

%% Perform Exemplar-SVM training
train_params = params;
train_params.detect_max_scale = 0.5;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;
train_params.train_max_negatives_in_cache = 5000;
train_params.train_max_mined_images = 500;

models = esvm_initialize_dt(pos_set, params);
models = esvm_train_exemplars(models, neg_set, train_params);

for niter = 1:5
  models = esvm_latent_update_dt(models, train_params);
  models = esvm_train_exemplars(models, neg_set, train_params);
end

% val_params = params;
% val_params.detect_exemplar_nms_os_threshold = 0.5;
% val_params.gt_function = @esvm_load_gt_function;
% val_set_name = ['trainval'];
% val_set = esvm_get_pascal_set(dataset_params, val_set_name);

%% Apply trained exemplars on validation set
%val_grid = esvm_detect_imageset(val_set, models, val_params, val_set_name);
                       
%% Perform Platt calibration and M-matrix estimation
%M = esvm_perform_calibration(val_grid, val_set, models,
%val_params);

%% Define test-set
test_params = params;
test_params.detect_exemplar_nms_os_threshold = 0.5;
test_set_name = ['test'];
test_set = esvm_get_pascal_set(dataset_params, test_set_name);

%% Apply on test set
test_grid = esvm_detect_imageset(test_set, models, test_params, test_set_name);

%% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, [], ...
                                      test_params);

%% Perform the exemplar evaluation
[results] = esvm_evaluate_pascal_voc(test_struct, test_grid, params, ...
                                     test_set_name, cls, ...
                                     models_name);



for mind = 1:length(models)
  I = esvm_show_top_exemplar_dets(test_struct, test_set, ...
                                  models, mind,10,10);
  filer = sprintf('%s/www/aicon.%s-%d.png',...
                  results_directory, models_name, mind);
filer
  imwrite(I,filer);
  
end

return;

%NOTE: the show_top_dets functions doesn't even work for dalaltriggs (but we can apply NN
%hack! if we want to)


%% Show top 20 detections as exemplar-inpainting results
maxk = 20;
allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
                            params,  maxk, test_set_name);



