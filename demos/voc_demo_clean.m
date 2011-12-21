function voc_demo_clean
%This is the main Exemplar-SVM driver program, see README.md for
%instructions
%Tomasz Malisiewicz (tomasz@cmu.edu)

%single exemplar training

cls = 'bus';

%% Initialize dataset
dataset_params = get_voc_dataset('VOC2007');
dataset_params.display = 0;

%Do not skip evaluation, unless it is VOC2010
dataset_params.SKIP_EVAL = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SET SOURCETRAIN/VAL/TEST PARAMS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize framing function
init_params.sbin = 8;
init_params.goal_ncells = 100;
init_params.MAXDIM = 12;
init_params.init_function = @initialize_goalsize_model;
%init_params.init_string = 'g';
init_params.init_type = 'g'; %sprintf('%s-%d-%d',...
                            %    init_params.init_string,...
                            %init_params.goal_ncells,...
                            %    init_params.MAXDIM);

dataset_params.init_params = init_params;

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 1;
dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%% Initialize exemplars with the exemplar stream
e_stream_set = get_pascal_stream(dataset_params, cls);


%Create mining/validation/testing params as defaults
dataset_params.params = get_default_mining_params;

%Choose the training function (do_svm, do_rank, ...)
%Disable NMS in training params
dataset_params.mining_params = dataset_params.params;
dataset_params.mining_params.training_function = @esvm_update_svm;
dataset_params.mining_params.NMS_OS = 1.0;
dataset_params.mining_params.MAXSCALE = 0.5;
dataset_params.mining_params.TOPK = 100;
dataset_params.mining_params.set_name = ['train-' cls];
%optional cap
%dataset_params.mining_params.set_maxk = 0;

dataset_params.val_params = dataset_params.params;
dataset_params.val_params.NMS_OS = 0.5;
dataset_params.val_params.set_name = ['trainval+' cls];;
%optional cap
%dataset_params.val_params.set_maxk = 10000;

dataset_params.test_params = dataset_params.params;
dataset_params.test_params.NMS_OS = 0.5;
dataset_params.test_params.set_name = ['test+' cls];
%optional cap
%dataset_params.test_params.set_maxk = 0;

%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' ...
     dataset_params.model_type];

%choose all classes except person
%classes = dataset_params.classes;
%classes = setdiff(classes,{'person'});

dataset_params.classes = {cls};
  
neg_set = get_pascal_set(dataset_params, ...
                           dataset_params.mining_params.set_name);
neg_set = neg_set(1:10);

val_set = get_pascal_set(dataset_params, ...
                         dataset_params.val_params.set_name);
val_set = val_set(1:10);

test_set = get_pascal_set(dataset_params, ...
                          dataset_params.test_params.set_name);
test_set = test_set(1:20);


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
%dataset_params.SKIP_M = 1;

voc_template(dataset_params, e_stream_set, neg_set, val_set, ...
             test_set, cls);

