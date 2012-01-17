%This is an Exemplar-SVM training demo
%Tomasz Malisiewicz (tomasz@cmu.edu)
function [models,M] = esvm_demo_train_synthetic

%% Create a synthetic dataset of cirlces
Npos = 1;
Nneg = 10;
[positive_stream,neg_set] = esvm_generate_dataset(Npos,Nneg);
cls = positive_stream{1}.cls;
if ~exist('results_directory','var')
  results_directory = sprintf(['/nfs/baikal/tmalisie/synthetic/']);
end

% Issue warning if lock files are present
lockfiles = check_for_lock_files(results_directory);
if length(lockfiles) > 0
  fprintf(1,'WARNING: %d lockfiles present in current directory\n', ...
          length(lockfiles));
end

%KILL_LOCKS = 1;
%for i = 1:length(lockfiles)
%  unix(sprintf('rmdir %s',lockfiles{i}));
%end

%% Set exemplar-initialization parameters
params = esvm_get_default_params;
params.model_type = 'exemplar';
%if next line is commented, no saving is performed
%params.dataset_params.localdir = results_directory;
params.dataset_params.display = 1;

e_stream_set = positive_stream;

%Choose a models name to indicate the type of training run we are
%doing.  If models_name is not specified, no saving is performed
%models_name = ...
%    [cls '-' params.init_params.init_type ...
%     '.' params.model_type];

initial_models = esvm_initialize_exemplars(e_stream_set, params);%, models_name);

%% Perform Exemplar-SVM training
train_params = params;
train_params.detect_max_scale = 0.5;
train_params.train_max_mined_images = 50;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;

% val_params = params;
% val_params.detect_exemplar_nms_os_threshold = 0.5;
% val_params.gt_function = @get_pascal_anno_function;
% val_params.CACHE_BETAS = 1;

% val_set_name = ['trainval+' cls];

% val_set = get_pascal_set(dataset_params, val_set_name);
% val_set = val_set(1:10);

%% Define test-set
test_params = params;
test_params.detect_exemplar_nms_os_threshold = 0.5;
test_set_name = ['test'];
[stream_test] = esvm_generate_dataset(10);
test_set = cellfun2(@(x)x.I,stream_test);

%% Train the exemplars and get updated models name
[models,models_name] = esvm_train_exemplars(initial_models, ...
                                            neg_set, train_params);


%% Apply trained exemplars on validation set
%val_grid = esvm_detect_imageset(val_set, models, val_params, val_set_name);

%% Perform Platt calibration and M-matrix estimation
%M = esvm_perform_calibration(val_grid, models, val_params);

%% Apply on test set
test_grid = esvm_detect_imageset(test_set, models, test_params);%, test_set_name);

%% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, [], test_params);

%% Show top detections
maxk = 20;
allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
                       params,  maxk, test_set_name);

