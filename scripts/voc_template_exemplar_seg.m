function voc_template_exemplar_seg(cls,VOCYEAR)
%Choose only objects which have segmentations!
if ~exist('VOCYEAR','var')
  VOCYEAR = 2007;
end

scenestring = 'exemplar';
stream_f = @(a,b,c,d)get_pascal_exemplar_stream(a,b,c,d,1);
get_default_param_f = @get_default_mining_params;
init_function = @initialize_fixedframe_model;

%Choose a short string to indicate the type of training run we are doing
models_name = ['fseg.' scenestring];

init_params.sbin = 8;
init_params.hg_size = [8 8];
init_params.goal_ncells = 100;

stream_set_name = 'trainval';
MAX_NUM_EX = 5;

training_function = @do_svm;

%Get the default mining parameters
mining_params = get_default_param_f();
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 0;
mining_params.dump_images = 0;
mining_params.MAXSCALE = 0.5;
mining_params.NMS_MINES_OS = 1.0;

mining_params.MAX_WINDOWS_BEFORE_SVM = 100;
mining_params.TOPK = 10;
mining_params.MAX_TOTAL_MINED_IMAGES = 20;

NIMS_PER_CHUNK = 4;

trainset_name = 'train';
trainset_name2 = ['-' cls];
trainset_maxk = 5;

valset_name = 'trainval';
valset_name2 = cls;
valset_maxk = 5;

val_gt_function = @get_pascal_anno_function;
val_params = get_default_param_f();

testset_name = 'test';
%% for 2010, we only have testset
%testset_name2 = '';
testset_name2 = cls;
testset_maxk = 20;
test_gt_function = [];%
%test_gt_function = @get_pascal_anno_function;
test_params = get_default_param_f();

%%%% SETUP DATASET
%CHOOSE HOW MANY IMAGES WE APPLY PER CHUNK
dataset_params.NIMS_PER_CHUNK = NIMS_PER_CHUNK;

%devkitroot is where we write all the result files
dataset_params.dataset = VOCYEAR;
dataset_params.testset = 'test';

dataset_params.devkitroot = ['/nfs/baikal/tmalisie/summer11/' ...
                    dataset_params.dataset];

% change this path to a writable local directory for the example code
dataset_params.localdir=[dataset_params.devkitroot '/local/'];

% change this path to a writable directory for your results
dataset_params.resdir=[dataset_params.devkitroot ['/' ...
                    'results/']];

%This is the directory where we dump visualizations into
[v,r] = unix('hostname');
if strfind(r,'airbone')==1
  dataset_params.datadir ='/projects/Pascal_VOC/';
  dataset_params.display_machine = 'airbone';
else
  dataset_params.datadir ='/nfs/hn38/users/sdivvala/Datasets/Pascal_VOC/';
  dataset_params.display_machine = 'onega';
end

dataset_params = VOCinit(dataset_params);


%%%% SETUP STREAMS

e_stream_set = stream_f(stream_set_name, cls, ...
                        dataset_params, MAX_NUM_EX);

test_set = get_pascal_bg(testset_name,testset_name2,dataset_params);
test_set = test_set(1:min(length(test_set),testset_maxk));

train_set = get_pascal_bg(trainset_name,trainset_name2,dataset_params);
train_set = train_set(1:min(length(train_set),trainset_maxk));

val_set = get_pascal_bg(valset_name,valset_name2,dataset_params);
val_set = val_set(1:min(length(val_set),valset_maxk));



%%%% MAIN PIPELINE

%Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(e_stream_set,...
                             init_function,init_params, ...
                             dataset_params,models_name);

dataset_params.display_machine = '';
%Load all of the initialized exemplars
models = load_all_models(cls,models_name,efiles,dataset_params,1);

%Train each exemplar
[tfiles, models_name] = train_all_exemplars(models, train_set, ...
                                            mining_params, dataset_params, ...
                                            training_function);

%Load the trained exemplars
models = load_all_models(cls,models_name,tfiles,dataset_params, 1);

%Apply exemplars on validation set
%val_files = apply_all_exemplars(models, dataset_params,...
%                    val_set, valset_name,...
%                    [], val_params, val_gt_function);

%Load validation results
%val_grid = load_result_grid(models, dataset_params, ...
%                            valset_name, val_files);

%perform calibration and M matrix estimation
%M = mmhtit(models, val_grid, dataset_params);

%Apply exemplars on test set
test_files = apply_all_exemplars(models, dataset_params,...
                    test_set, testset_name,...
                    [], test_params, test_gt_function);

%Load testset results
test_grid = load_result_grid(models, dataset_params, ...
                             testset_name, test_files);

%Evaluation of RAW SVM classifiers + DISPLAY
teststruct = pool_results(dataset_params,models,test_grid);
% [results] = evaluate_pascal_voc_grid(dataset_params, ...
%                                      models, test_grid, ...
%                                      testset_name, teststruct);
%show_top_dets(dataset_params, models, test_grid,...
%              test_set, testset_name, ...
%              teststruct);

%Evaluation of just Platt's calibration + DISPLAY
%M2.betas = M.betas;
%teststruct = pool_results(dataset_params,models,test_grid,M2);
% [results] = evaluate_pascal_voc_grid(dataset_params, ...
%                                      models, test_grid, ...
%                                      testset_name, teststruct);
%show_top_dets(dataset_params, models, test_grid,...
%              test_set, testset_name, ...
%              teststruct);

%Evaluation of Platt's calibration + M boosting + DISPLAY
%teststruct = pool_results(dataset_params,models,test_grid,M);

%% no evaluation for VOC2010 dataset.. need to do it online
%[results] = evaluate_pascal_voc_grid(dataset_params, ...
%                                     models, test_grid, ...
%                                     testset_name, teststruct);
show_top_dets(dataset_params, models, test_grid,...
              test_set, testset_name, ...
              teststruct);
