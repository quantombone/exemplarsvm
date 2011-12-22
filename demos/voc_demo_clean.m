%This is an Exemplar-SVM training demo
%Tomasz Malisiewicz (tomasz@cmu.edu)

%% Initialize dataset parameters
data_directory = '/Users/tomasz/projects/Pascal_VOC/';
results_directory = '/nfs/baikal/tmalisie/esvm-data/';
cls = 'bus';
dataset_params = get_voc_dataset('VOC2007',...
                                 data_directory,...
                                 results_directory);

%% Set exemplar-initialization parameters

%Initialize framing function
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 12;
init_params.init_function = @initialize_goalsize_model;
init_params.init_type = 'g'; 
dataset_params.init_params = init_params;

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 1;
dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%Create an exemplar stream (list of exemplars)
e_stream_set = get_pascal_stream(dataset_params, cls);

%% Define parameters and training
%Create mining/validation/testing params as defaults
dataset_params.params = get_default_mining_params;
dataset_params.mining_params = dataset_params.params;
dataset_params.mining_params.training_function = @esvm_update_svm;
dataset_params.mining_params.NMS_OS = 1.0; %disable NMS
dataset_params.mining_params.MAXSCALE = 0.5;
dataset_params.mining_params.TOPK = 100;
dataset_params.mining_params.set_name = ['train-' cls];
neg_set = get_pascal_set(dataset_params, ...
                           dataset_params.mining_params.set_name);
neg_set = neg_set(1:100);

%% Define validation set
dataset_params.val_params = dataset_params.params;
dataset_params.val_params.NMS_OS = 0.5;
dataset_params.val_params.set_name = ['trainval+' cls];;
val_set = get_pascal_set(dataset_params, ...
                         dataset_params.val_params.set_name);

%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' dataset_params.model_type];

%% Perform Exemplar-SVM training
[models, M] = esvm_train_and_calibrate(dataset_params, e_stream_set, ...
                                       neg_set, val_set, cls);

%% Define test-set
dataset_params.test_params = dataset_params.params;
dataset_params.test_params.NMS_OS = 0.5;
dataset_params.test_params.set_name = ['test+' cls];
test_set = get_pascal_set(dataset_params, ...
                          dataset_params.test_params.set_name);


%% Apply on test set
dataset_params.params = dataset_params.test_params;
test_files = esvm_detect_set(dataset_params, models, test_set, ...
                             dataset_params.test_params.set_name);

%Load test set results
test_grid = esvm_load_result_grid(dataset_params, models, ...
                                  dataset_params.test_params.set_name, ...
                                  test_files);

%apply calibration matrix to test-set results
test_struct = esvm_pool_exemplars(dataset_params, models, ...
                                  test_grid, M);

%Show top hits
bbs = cat(1,test_struct.unclipped_boxes{:});
[aa,bb] = sort(bbs(:,end),'descend');
bbs = bbs(bb,:);
m = models{1};
m.model.svbbs = bbs;
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


%maxk = 10;
%allbbs = show_top_dets(dataset_params, models, test_grid, test_set, dataset_par%ams.test_params.set_name, ...
%                         test_struct, maxk);
