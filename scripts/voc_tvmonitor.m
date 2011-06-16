clear

%Choose the VOC category
cls = 'tvmonitor';

%Choose a short string to indicate the type of training run we are doing
mode = 'f';

%get the exemplar stream from VOC
e_set = get_pascal_stream('trainval', cls);

%take only first 5 objects
e_set = e_set(1:5);

init_params.mode = mode;
init_params.SBIN = 8;
init_params.hg_size = [8 8];
init_params.topK = 1;
init_params.ADD_LR = 0;
init_function = @initialize_fixedframe_model;

% init_params.mode = mode;
% init_params.SBIN = 8;
% init_params.GOAL_NCELLS = 100;
% init_function = @initialize_goalsize_model;

%Initialize exemplars with the exemplar stream
exemplar_initialize(e_set,init_params,init_function);

%get the negative set for training
neg_set = get_pascal_bg('train',['-' cls]);
neg_set = neg_set(1:3);

%Get the default mining parameters
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 1;
mining_params.dump_images = 1;
mining_params.MAXSCALE = 0.5;
mining_params.FLIP_LR = 1;
mining_params.NMS_MINES_OS = 1.0;
mining_params.extract_negatives = 0;
mining_params.alternate_validation = 0;
mining_params.MAX_WINDOWS_BEFORE_SVM = 40;

train_all_exemplars(cls, mode, neg_set, mining_params);

return;
%get the application set for calibration
val_set = get_pascal_bg('trainval');

%get the testing set for evaluation
test_set = get_pascal_bg('test');
