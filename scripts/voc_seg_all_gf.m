error('deprecated, must reflect changes in voc_demo_nn');
clear;

%% Initialize dataset
VOCYEAR = 'VOC2007';
suffix = '/nfs/baikal/tmalisie/demo11/';
dataset_params = get_voc_dataset(VOCYEAR,suffix);
dataset_params.display = 0;

dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
%dataset_params.model_type = 'scene';
dataset_params.model_type = 'exemplar';

%Initialize framing function
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 12;
init_params.init_function = @initialize_goalsize_model;
init_params.init_type = sprintf('g-%d-%d',...
                                init_params.goal_ncells,...
                                init_params.MAXDIM);

% %Initialize framing function
% init_params.sbin = 8;
% init_params.hg_size = [8 8];
% init_params.init_function = @initialize_fixedframe_model;
% init_params.init_type = sprintf('f-%d-%d',...
%                                 init_params.hg_size(1),...
%                                 init_params.hg_size(2));

dataset_params.init_params = init_params;

dataset_params.mining_params = dataset_params.params;

%Choose the training function (do_svm, do_rank, ...)
dataset_params.mining_params.training_function = @do_svm;

%Disable NMS for training
dataset_params.mining_params.NMS_OS = 1.0;
dataset_params.mining_params.MAXSCALE = 0.5;
dataset_params.mining_params.MAX_WINDOWS_BEFORE_SVM = 1000;
dataset_params.mining_params.TOPK = 50;
dataset_params.mining_params.MAX_TOTAL_MINED_IMAGES = 2000;

%Get the default mining parameters (plus some fixes for training)
mining_params = get_default_mining_params;

dataset_params.mining_params = mining_params;
dataset_params.val_params = dataset_params.params;
dataset_params.test_params = dataset_params.params;

if strcmp(dataset_params.model_type, 'scene')
  dataset_params.val_params.MIN_SCENE_OS = 0.5;
  dataset_params.test_params.MIN_SCENE_OS = 0.5;
end

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 2000;

dataset_params.trainset_name = 'train';
dataset_params.trainset_maxk = 2000;

dataset_params.valset_name = 'trainval';
dataset_params.valset_maxk = 10000;

dataset_params.testset_name = 'test';
dataset_params.testset_maxk = 10000;

%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' ...
     dataset_params.model_type];

dataset_params.SKIP_TRAINING = 0;
dataset_params.SKIP_VAL = 0;
dataset_params.SKIP_EVAL = 0;

classes = {...
    'dog'
    'cat'
    'tvmonitor'
    'bicycle'
    'cow'
    'motorbike'
    'dog'
    'sofa'
    'boat'
};

myRandomize;
r = randperm(length(classes));
classes = classes(r);

for i = 1:length(classes)
  %Training set is negatives
  dataset_params.trainset_name2 = ['-' classes{i}];
  %Validation set is all of trainval
  dataset_params.valset_name2 = '';%['-' classes{i}];
  %Test-set is all of testing
  dataset_params.testset_name2 = '';%classes{i};
  
  voc_template(dataset_params, classes{i});
end
