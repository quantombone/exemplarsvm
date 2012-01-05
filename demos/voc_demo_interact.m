%This is an Exemplar-SVM training demo
%Tomasz Malisiewicz (tomasz@cmu.edu)
function voc_demo_interact

%% Initialize dataset parameters
data_directory = '/Users/tmalisie/projects/pascal/VOCdevkit/';
%results_directory = '/nfs/baikal/tmalisie/esvm-data/';


%results_directory = '/nfs/baikal/tmalisie/esvm-car/';

%data_directory = '/csail/vision-videolabelme/people/tomasz/VOCdevkit/';
%results_directory = sprintf('/csail/vision-videolabelme/people/tomasz/esvm-%s/',cls);

dataset_parms.params = get_default_mining_params;
%dataset_params
dataset_params = get_voc_dataset('VOC2007',...
                                 data_directory);%,...
%                                 results_directory);


%% Set exemplar-initialization parameters

%Initialize framing function
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 12;
init_params.init_function = @esvm_initialize_goalsize_exemplar;
init_params.init_type = 'g'; 
dataset_params.init_params = init_params;



%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 5000;
dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%fields of e_stream_set{1}
%            I: [1x71 char]
%         bbox: [43 66 279 363]
%          cls: 'bus'
%     objectid: 1
%        curid: '000132'
%         anno: [1x1 struct]
%        filer: '000132.1.bus.mat'


% sub = initialize_screenshot;
% es.I = capture_screen(sub);
% es.bbox = select_bbox_from_image(es.I, 'select exemplar');
% es.cls = 'selection';
% e_stream_set{1} = es;


load e_stream_set.mat

% save e_stream_set.mat e_stream_set

%% Define parameters and training
%Create mining/validation/testing params as defaults
dataset_params.params = get_default_mining_params;
dataset_params.mining_params = dataset_params.params;
dataset_params.mining_params.training_function = @esvm_update_svm;
dataset_params.mining_params.NMS_OS = 1.0; %disable NMS
dataset_params.mining_params.MAXSCALE = 0.5;
dataset_params.mining_params.TOPK = 100;
dataset_params.mining_params.set_name = ['trainval'];
neg_set = get_pascal_set(dataset_params, ...
                           dataset_params.mining_params.set_name);
neg_set = neg_set(20);

%Choose a models name to indicate the type of training run we are doing
dataset_params.models_name = ...
    ['selection' '-' init_params.init_type ...
     '.' dataset_params.model_type];

%% Exemplar initialization 
CACHE_MODELS = 0;
initial_models = esvm_initialize_exemplars(dataset_params, e_stream_set, ...
                                           dataset_params.init_params, ...
                                           dataset_params.models_name,...
                                           CACHE_MODELS);

%Append the nn-type string if we are in nn mode
if length(dataset_params.params.nnmode) > 0
  models_name = [models_name '-' dataset_params.params.nnmode];
end

%% Perform Exemplar-SVM training
CACHE_MODELS = 1;
models = esvm_train_exemplars(dataset_params, ...
                                initial_models, neg_set, CACHE_MODELS);
 
%% Apply trained exemplars on validation set
dataset_params.params = dataset_params.val_params;
dataset_params.val_params.gt_function = @get_pascal_anno_function;
val_grid = esvm_detect_imageset(val_set,models,...
                                dataset_params.val_params,...
                                dataset_params.val_params.set_name,...
                                dataset_params);

%% Perform Platt calibration and M-matrix estimation
CACHE_BETAS = 1;
M = esvm_perform_calibration(dataset_params, models, ...
                             val_grid, val_set, CACHE_BETAS);

%% Define test-set
dataset_params.test_params = dataset_params.params;
dataset_params.test_params.NMS_OS = 0.5;
dataset_params.test_params.set_name = ['test'];
test_set = get_pascal_set(dataset_params, ...
                          dataset_params.test_params.set_name);

%% Apply on test set
dataset_params.params = dataset_params.test_params;
test_grid = esvm_detect_imageset(test_set, models, ...
                                 dataset_params.test_params,...
                                 dataset_params.test_params.set_name, ...
                                 dataset_params);

%apply calibration matrix to test-set results
test_struct = esvm_apply_calibration(dataset_params, models, ...
                                     test_grid, M);

%Show top hits
bbs = cat(1,test_struct.unclipped_boxes{:});
[aa,bb] = sort(bbs(:,end),'descend');
bbs = bbs(bb,:);
m = models{1};
m.model.svbbs = bbs;
try
m.model = rmfield(m.model,'svxs');
catch
end
m.train_set = test_set;

figure(1)
clf
imagesc(get_sv_stack(m,4,4))
axis image
axis off
title('Exemplar, w,  and top 16 detections');

[results] = evaluate_pascal_voc_grid(dataset_params, ...
                                     models, test_grid, ...
                                     dataset_params.test_params.set_name, ...
                                     test_struct);
%rc = results.corr;
%clear options
%options.format ='html';
%options.outputDir = [results_directory  '/www/'];
%publish('display_helper',options)

%if enabled show and print some top detections into the www directory
%maxk = 2;
%allbbs = show_top_dets(dataset_params, models, test_grid, ...
%                       test_set, dataset_params.test_params.set_name, ...
%                       test_struct, maxk);
