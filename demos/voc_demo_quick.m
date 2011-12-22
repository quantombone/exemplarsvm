function voc_demo_quick
%In this quick demo, we perform the following
% 1) Choose one exemplar (a bus from PASCAL VOC 2007 trainval set)
% 2) Define datasets:
%       negative set as 10 random non-bus images from trainval
%       validation set as 20 random bus images from trainval
%       test-set as 20 random bus images from test-set
% 5) Perform esvm training, calibration on validation set, and
% evaluation on test-set
% 6) Show an example AP curve
%Tomasz Malisiewicz (tomasz@cmu.edu)

%Choose the class to work with
cls = 'bus';

%% Initialize dataset
dataset_params = get_voc_dataset('VOC2007');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET SOURCETRAIN/VAL/TEST PARAMS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize framing function
dataset_params.init_params.sbin = 8;
dataset_params.init_params.goal_ncells = 100;
dataset_params.init_params.MAXDIM = 12;
dataset_params.init_params.init_function = @initialize_goalsize_model;
dataset_params.init_params.init_string = 'g';
dataset_params.init_params.init_type = ...
    sprintf('%s-%d-%d',...
            dataset_params.init_params.init_string,...
            dataset_params.init_params.goal_ncells,...
            dataset_params.init_params.MAXDIM);

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 5000;

%% Initialize exemplars with the exemplar stream
e_stream_set = get_pascal_stream(dataset_params, cls);

dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%Create mining/validation/testing params as defaults
dataset_params.params = get_default_mining_params;

%Choose the training function (do_svm, do_rank, ...)
%Disable NMS in training params
dataset_params.mining_params = dataset_params.params;
dataset_params.mining_params.training_function = @do_svm;
dataset_params.mining_params.NMS_OS = 1.0;
dataset_params.mining_params.MAXSCALE = 0.5;
dataset_params.mining_params.TOPK = 100;
dataset_params.mining_params.set_name = 'train';

dataset_params.val_params = dataset_params.params;
dataset_params.val_params.NMS_OS = 0.5;
dataset_params.val_params.set_name = 'trainval';

dataset_params.test_params = dataset_params.params;
dataset_params.test_params.NMS_OS = 0.5;
dataset_params.test_params.set_name = 'test';

%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' ...
     dataset_params.model_type];
  
%Training set is images not containing in-class instances
dataset_params.mining_params.set_name = ...
    [dataset_params.mining_params.set_name '-' cls];
train_set = get_pascal_set(dataset_params, ...
                           dataset_params.mining_params.set_name);

val_set = get_pascal_set(dataset_params, ...
                         dataset_params.val_params.set_name);

test_set = get_pascal_set(dataset_params, ...
                          dataset_params.test_params.set_name);

%dataset_params.JUST_TRAIN = 1;
%dataset_params.JUST_TRAIN_AND_LOAD = 1;
%dataset_params.SKIP_M = 1;

voc_template(dataset_params, e_stream_set, ...
             neg_set, val_set, ...
             test_set, cls);

