%clear;

%% Initialize dataset
VOCYEAR = 'VOC2007';
suffix = '/nfs/baikal/tmalisie/nn311/';
dataset_params = get_voc_dataset(VOCYEAR,suffix);
dataset_params.display = 0;

%Do not skip evaluation, unless it is VOC2010
dataset_params.SKIP_EVAL = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET SOURCETRAIN/VAL/TEST PARAMS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1
  %Initialize framing function
  init_params.sbin = 8;
  init_params.goal_ncells = 100;
  init_params.MAXDIM = 12;
  init_params.init_function = @initialize_goalsize_model;
  init_params.init_string = 'g';
  init_params.init_type = sprintf('%s-%d-%d',...
                                  init_params.init_string,...
                                  init_params.goal_ncells,...
                                  init_params.MAXDIM);
else
  %Initialize exemplar framing function
  init_params.sbin = 8;
  init_params.hg_size = [8 8];
  init_params.init_function = @initialize_fixedframe_model;
  init_params.init_string = 'f';
  init_params.init_type = sprintf('%s-%d-%d', ...
                                  init_params.init_string, ...
                                  init_params.hg_size(1), ...
                                  init_params.hg_size(2));
end
dataset_params.init_params = init_params;

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 5000;

dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%Create mining/validation/testing params as defaults
dataset_params.params = get_default_mining_params;
%dataset_params.params.nnmode = 'normalizedhog';

if 1
  %Choose the training function (do_svm, do_rank, ...)
  %Disable NMS in training params
  dataset_params.mining_params = dataset_params.params;
  dataset_params.mining_params.training_function = @do_svm;
  dataset_params.mining_params.NMS_OS = 1.0;
  dataset_params.mining_params.MAXSCALE = 0.5;
  dataset_params.mining_params.TOPK = 100;
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

%classes = {'bus'};
classes = {...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'cat'
    'car'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    %'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'
};

%classes = {'bicycle'};
%classes = {'person'};
%classes = {'train'};
%classes = {'bicycle'};
%classes = {'motorbike'};
%classes = {'sheep'};
%classes = {'person'};
%classes = {'bus'};

%classes = {'motorbike'};
classes = {'sheep','cow'};
myRandomize;
r = randperm(length(classes));
classes = classes(r);

%plot_voc_results_horiz(dataset_params);
%
%return;
save_dataset_params = dataset_params;
for i = 1:length(classes)
  dataset_params = save_dataset_params;
  
   if isfield(dataset_params,'mining_params')
     %Training set is images not containing in-class instances
     dataset_params.mining_params.set_name = ...
         [dataset_params.mining_params.set_name '-' classes{i}];
   end

  % if isfield(dataset_params,'val_params')
  %   %Validate on in-class images only
  %   dataset_params.val_params.set_name = ...
  %       [dataset_params.val_params.set_name '+' classes{i}];
  % end

  % if isfield(dataset_params,'test_params')
  %   %Test on in-class images only
  %   dataset_params.test_params.set_name = ...
  %       [dataset_params.test_params.set_name '+' classes{i}];
  % end

  %dataset_params.JUST_TRAIN = 1;
  %dataset_params.JUST_TRAIN_AND_LOAD = 1;
  dataset_params.SKIP_M = 1;
  NNN = 20;
  voc_template(dataset_params, classes{i}, NNN);
end
