%clear

%Choose the VOC category
cls = 'car';

%Choose a short string to indicate the type of training run we are doing
models_name = 'fscene';

%Choose Fixed-Frame initialization function, and its parameters
%init_function = @initialize_fixedframe_model;
%init_params.sbin = 8;
%init_params.hg_size = [8 8];

%Choose Goal-Cells initialize function, and its parameters
init_function = @initialize_goalsize_model;
init_params.sbin = 8;
init_params.goal_ncells = 100;

%devkitroot is where we write all the result files
dataset_params.dataset = 'VOC2007';
dataset_params.testset = 'test';
dataset_params.devkitroot = ['/nfs/baikal/tmalisie/summer11/' dataset_params.dataset];;
dataset_params.wwwdir = [dataset_params.devkitroot '/www/'];

%This is the directory where we dump visualizations into
[v,r] = unix('hostname');
if strfind(r,'airbone')==1
  dataset_params.datadir ='/projects/Pascal_VOC/';
  dataset_params.display_machine = 'airbone';
else
  dataset_params.datadir ='/nfs/hn38/users/sdivvala/Datasets/Pascal_VOC/';
  dataset_params.display_machine = 'onega';
end

%dataset_params.dataset='VOC2010';


% change this path to a writable local directory for the example code
dataset_params.localdir=[dataset_params.devkitroot '/local/'];

% change this path to a writable directory for your results
dataset_params.resdir=[dataset_params.devkitroot ['/' ...
                    'results/']];

dataset_params = VOCinit(dataset_params);


%get the exemplar stream from VOC
stream_set_name = 'trainval';
MAX_NUM_EX = 20;
e_stream_set = get_pascal_scene_stream(stream_set_name, cls, dataset_params, MAX_NUM_EX);

%Initialize exemplars with the exemplar stream
exemplar_initialize(e_stream_set,init_function,init_params, ...
                    dataset_params,models_name);

models = load_all_models(cls,models_name,dataset_params,1);

%get the negative set for training
train_set = get_pascal_bg('train',['-' cls],dataset_params);
train_set = train_set(1:5);

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 0;%1;
mining_params.dump_images = 0;%1;
mining_params.MAXSCALE = 0.5;
mining_params.FLIP_LR = 1;
mining_params.NMS_MINES_OS = 1.0;
mining_params.extract_negatives = 0;
mining_params.alternate_validation = 0;
mining_params.MAX_WINDOWS_BEFORE_SVM = 100;
mining_params.TOPK = 10;
mining_params.MAX_TOTAL_MINED_IMAGES = 20;

training_function = @do_svm;
train_all_exemplars(models, train_set, mining_params, ...
                    dataset_params, training_function);

models_name = [models_name '-svm'];

%Load the trained outputs
models = load_all_models(cls,models_name,dataset_params, 1);

curset_name = 'trainval';
val_set = get_pascal_bg(curset_name,cls,dataset_params);
val_set = val_set(1:20);

dataset_params.display_machine = '';
%No calibration parameters yet

M = [];
%CHOOSE HOW MANY IMAGES WE APPLY PER CHUNK
dataset_params.NIMS_PER_CHUNK = 10;
%gt_function = @get_pascal_anno_function;

test_params = get_default_mining_params;
test_params.MAXSCALE = .4;
test_params.MIN_SCENE_OS = 0.4;
apply_all_exemplars(models,dataset_params,...
                    val_set,curset_name,...
                    M,test_params);

grid = load_result_grid(models, dataset_params, curset_name);

[results,finalstruct] = evaluate_pascal_voc_grid(dataset_params, models, ...
                                           grid, curset_name, M);

%Show bbs and save them into the www dir
allbbs = show_top_dets(dataset_params, models, grid, val_set, ...
                       finalstruct);

return;
%get the application set for calibration

%Get the testing set for evaluation
test_set = get_pascal_bg('test','',dataset_params);
