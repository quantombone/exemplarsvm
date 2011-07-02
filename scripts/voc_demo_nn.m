clear;

%% Initialize dataset
VOCYEAR = 'VOC2007';
suffix = '/nfs/baikal/tmalisie/demo11/';
dataset_params = get_voc_dataset(VOCYEAR,suffix);
dataset_params.display = 0;

dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%Initialize framing function
% init_params.sbin = 8;
% init_params.goal_ncells = 100;
% init_params.MAXDIM = 12;
% init_params.init_function = @initialize_goalsize_model;
% init_params.init_type = sprintf('nng-%d-%d',...
%                                 init_params.goal_ncells,...
%                                 init_params.MAXDIM);

%Initialize framing function
init_params.sbin = 8;
init_params.hg_size = [8 8];
init_params.init_function = @initialize_fixedframe_model;
init_params.nnmode = 1;
init_params.init_type = sprintf('nn-%d-f-%d-%d', ...
                                init_params.nnmode, ...
                                init_params.hg_size(1), ...
                                init_params.hg_size(2));

%Choose the training function (do_svm, do_rank, ...)
dataset_params.training_function = @do_svm;

dataset_params.init_params = init_params;

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 100;
dataset_params.trainset_name = 'train';
dataset_params.trainset_maxk = 0;

dataset_params.valset_name = 'trainval';
dataset_params.valset_maxk = 0;

dataset_params.testset_name = 'test';
dataset_params.testset_maxk = 1000;

dataset_params.mining_params = get_default_mining_params;

%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' ...
     dataset_params.model_type];

dataset_params.SKIP_TRAINING = 1;
dataset_params.SKIP_VAL = 1;
dataset_params.SKIP_EVAL = 1;

classes = {...
    'bus'
};

myRandomize;
r = randperm(length(classes));
classes = classes(r);

for i = 1:length(classes)
  %Training set is negatives
  dataset_params.trainset_name2 = ['-' classes{i}];
  
  %Validation set is in-class valset
  dataset_params.valset_name2 = [classes{i}];
  
  %Test-set is in-class testset
  dataset_params.testset_name2 = classes{i};
  
  voc_template(dataset_params, classes{i});
end
