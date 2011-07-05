clear;

%% Initialize dataset
VOCYEAR = 'VOC2007';
suffix = '/nfs/baikal/tmalisie/nn11/';
dataset_params = get_voc_dataset(VOCYEAR,suffix);
dataset_params.display = 0;

%SKIP_TRAINING = 1;
%SKIP_VAL = 0;

dataset_params.SKIP_EVAL = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET SOURCETRAIN/VAL/TEST PARAMS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize framing function
% init_params.sbin = 8;
% init_params.goal_ncells = 100;
% init_params.MAXDIM = 12;
% init_params.init_function = @initialize_goalsize_model;
% init_params.init_type = sprintf('nng-%d-%d',...
%                                 init_params.goal_ncells,...
%                                 init_params.MAXDIM);

%Initialize exemplar framing function
init_params.sbin = 8;
init_params.hg_size = [8 8];
init_params.init_function = @initialize_fixedframe_model;
init_params.init_string = 'f';
init_params.nnmode = 1;
init_params.nn_string = 'nn-1';
init_params.init_type = sprintf('%s-%s-%d-%d', ...
                                init_params.nn_string, ...
                                init_params.init_string, ...
                                init_params.hg_size(1), ...
                                init_params.hg_size(2));

dataset_params.init_params = init_params;

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 2000;

dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%Create mining/validation/testing params as defaults
dataset_params.params = get_default_mining_params;

if 0
  %Choose the training function (do_svm, do_rank, ...)
  %Disable NMS in training params
  dataset_params.mining_params = dataset_params.params;
  dataset_params.mining_params.training_function = @do_svm;
  dataset_params.mining_params.NMS_OS = 1.0;
  dataset_params.mining_params.MAXSCALE = 0.5;
  dataset_params.mining_params.set_name = 'train';
  %optional cap
  %dataset_params.mining_params.set_maxk = 0;
end

if 1
  dataset_params.val_params = dataset_params.params;
  dataset_params.val_params.NMS_OS = 0.5;
  dataset_params.val_params.set_name = 'trainval';
  %optional cap
  %dataset_params.val_params.set_maxk = 10000;
end

if 1
  dataset_params.test_params = dataset_params.params;
  dataset_params.test_params.NMS_OS = 0.5;
  dataset_params.test_params.set_name = 'test';
  %optional cap
  %dataset_params.test_params.set_maxk = 0;
end
%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' ...
     dataset_params.model_type];

classes = {'bus'};
% classes = {...
%     'aeroplane'
%     'bicycle'
%     'bird'
%     'boat'
%     'bottle'
%     'bus'
%     'car'
%     'cat'
%     'chair'
%     'cow'
%     'diningtable'
%     'dog'
%     'horse'
%     'motorbike'
%     'pottedplant'
%     'sheep'
%     'sofa'
%     'train'
%     'tvmonitor'
% };

myRandomize;
r = randperm(length(classes));
classes = classes(r);

for i = 1:length(classes)
  %Training set is negatives
  if isfield(dataset_params,'mining_params')
    dataset_params.mining_params.set_name2 = ['-' classes{i}];
  end

  if isfield(dataset_params,'val_params')
    dataset_params.val_params.set_name2 = '';
    %Validate on in-class images only
    dataset_params.val_params.set_name2 = classes{i};;
  end

  if isfield(dataset_params,'test_params')
    dataset_params.test_params.set_name2 = '';
    %Test on in-class images only
    dataset_params.test_params.set_name2 = classes{i};;
  end
    
  voc_template(dataset_params, classes{i});
end
