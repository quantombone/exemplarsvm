%This is an Exemplar-SVM training demo
%Tomasz Malisiewicz (tomasz@cmu.edu)
function voc_demo_esvm_train_fast(cls)

%% Initialize dataset parameters
%data_directory = '/Users/tomasz/projects/Pascal_VOC/';
%results_directory = '/nfs/baikal/tmalisie/esvm-data/';
if ~exist('cls','var')
  cls = 'car';
end

data_directory = '/Users/tmalisie/projects/pascal/VOCdevkit/';
results_directory = '/nfs/baikal/tmalisie/esvm-car/';

%data_directory = '/csail/vision-videolabelme/people/tomasz/VOCdevkit/';
%results_directory = sprintf('/csail/vision-videolabelme/people/tomasz/esvm-%s/',cls);

dataset_params = get_voc_dataset('VOC2007',...
                                 data_directory,...
                                 results_directory);

%% Issue warning if lock files are present
lockfiles = check_for_lock_files(results_directory);
if length(lockfiles) > 0
  fprintf(1,'WARNING: %d lockfiles present in current directory\n', ...
          length(lockfiles));
end

KILL_LOCKS = 1;
for i = 1:length(lockfiles)
  unix(sprintf('rmdir %s',lockfiles{i}));
end

%% Set exemplar-initialization parameters
params = esvm_get_default_params;
params.model_type = 'exemplar';
params.dataset_params = dataset_params;

%Initialize exemplar stream
stream_params.stream_set_name = 'trainval';
stream_params.stream_max_ex = 20;
stream_params.must_have_seg = 0;
stream_params.must_have_seg_string = '';
stream_params.model_type = 'exemplar'; %must be scene or exemplar;
stream_params.cache_file = 1;
stream_params.cls = cls;

%Create an exemplar stream (list of exemplars)
e_stream_set = esvm_get_pascal_stream(stream_params, dataset_params);

neg_set = get_pascal_set(dataset_params, ['train-' cls]);

%Choose a models name to indicate the type of training run we are doing
models_name = ...
    [cls '-' params.init_params.init_type ...
     '.' params.model_type];

initial_models = esvm_initialize_exemplars(e_stream_set, params, models_name);

%% Perform Exemplar-SVM training
train_params = params;
train_params.detect_max_scale = 0.5;
train_params.train_max_mined_images = 300;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;
train_params.CACHE_FILE = 1;

val_params = params;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.gt_function = @get_pascal_anno_function;
val_params.CACHE_BETAS = 1;

val_set_name = ['trainval+' cls];

val_set = get_pascal_set(dataset_params, val_set_name);
%val_set = val_set(1:50);

%% Define test-set
test_params = params;
test_params.detect_exemplar_nms_os_threshold = 0.5;
test_set_name = ['test+' cls];
test_set = get_pascal_set(dataset_params, test_set_name);
test_set = test_set(1);

%% Train the exemplars and get updated models name
[models,models_name] = esvm_train_exemplars(initial_models, ...
                                            neg_set, train_params);

%% Apply trained exemplars on validation set
val_grid = esvm_detect_imageset(val_set, models, val_params, val_set_name);
                       
%% Perform Platt calibration and M-matrix estimation
M = esvm_perform_calibration(val_grid, models, val_params);

%% Apply on test set
%test_params.max_models_before_block_method = 0;
test_grid = esvm_detect_imageset(test_set, models, test_params, test_set_name);

%% Apply calibration matrix to test-set results
test_struct = esvm_apply_calibration(test_grid, models, M, test_params);

% %% Show top hits for exemplar 1
% I = show_top_match_image(test_struct, test_set, models, 1);
% figure(1)
% clf
% imagesc(I)
% axis image
% axis off
% title('Exemplar, w,  and top 16 detections');

maxk = 2;
allbbs = show_top_dets(test_struct, test_grid, test_set, models, ...
                       params,  maxk, test_set_name);

%return
%[results] = evaluate_pascal_voc_grid(test_struct, test_grid,  ...
%                                     params, test_set_name, cls,
%                                     models_name);


%rc = results.corr;
%clear options
%options.format ='html';
%options.outputDir = [results_directory  '/www/'];
%publish('display_helper',options)

%if enabled show and print some top detections into the www directory
