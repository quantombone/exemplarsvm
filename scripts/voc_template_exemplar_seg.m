function voc_template_exemplar_seg(cls, VOCYEAR)
%In this script, we choose only objects which have segmentations!

if ~exist('VOCYEAR','var')
  VOCYEAR = 'VOC2007';
end

%Initialize dataset
dataset_params = get_voc_dataset(VOCYEAR);
dataset_params.display_machine = '';

%Choose a short string to indicate the type of training run we are doing
models_name = ['fseg.exemplar'];

%Initialize exemplar stream
stream_set_name = 'trainval';
stream_max_ex = 2000;
must_have_seg = 1;
e_stream_set = get_pascal_exemplar_stream(dataset_params, ...
                                          stream_set_name, ...
                                          cls, stream_max_ex,...
                                          must_have_seg);

%Initialize framing function
init_params.sbin = 8;
init_params.hg_size = [8 8];
init_params.goal_ncells = 100;
init_function = @(a,b)initialize_fixedframe_model(a,b,init_params);

%Choose the training function (do_svm, do_rank, ...)
training_function = @do_svm;

%Get the default mining parameters (plus some fixes for training)
mining_params = get_default_mining_params;
mining_params.SKIP_GTS_ABOVE_THIS_OS = 1.0;
mining_params.dump_last_image = 0;
mining_params.dump_images = 0;
mining_params.MAXSCALE = 0.5;
mining_params.NMS_MINES_OS = 1.0;
mining_params.MAX_WINDOWS_BEFORE_SVM = 1000;
mining_params.TOPK = 50;
mining_params.MAX_TOTAL_MINED_IMAGES = 2000;

trainset_name = 'train';
trainset_name2 = ['-' cls];
trainset_maxk = 50000;
train_set = get_pascal_set(dataset_params, trainset_name, trainset_name2);
train_set = train_set(1:min(length(train_set), trainset_maxk));

valset_name = 'trainval';
valset_name2 = 'train';
valset_maxk = 50000;
val_set = get_pascal_set(dataset_params, valset_name, valset_name2);
val_set = val_set(1:min(length(val_set),valset_maxk));
val_gt_function = @get_pascal_anno_function;
val_params = get_default_mining_params;

testset_name = 'test';
testset_name2 = '';
testset_maxk = 50000;
test_set = get_pascal_set(dataset_params, testset_name, testset_name2);
test_set = test_set(1:min(length(test_set),testset_maxk));
test_gt_function = [];
%test_gt_function = @get_pascal_anno_function;
test_params = get_default_mining_params;

if length(test_set) == 0
  fprintf(1,'Warning, testset is empty\n');
  return;
end

%%%% MAIN PIPELINE

%Initialize exemplars with the exemplar stream
efiles = exemplar_initialize(dataset_params, e_stream_set, ...
                             models_name, init_function);

%Load all of the initialized exemplars
models = load_all_models(dataset_params, cls, models_name, efiles, 1);


%Train each exemplar
[tfiles, models_name] = train_all_exemplars(dataset_params, models, ...
                                            train_set, mining_params, ...
                                            training_function);

%Load the trained exemplars
models = load_all_models(dataset_params, cls, models_name, tfiles, 1);

%Apply exemplars on validation set
val_files = apply_all_exemplars(dataset_params, models, val_set, ...
                                valset_name, [], val_params, val_gt_function);

%Load validation results
val_grid = load_result_grid(dataset_params, models, valset_name, val_files);

%perform calibration and M matrix estimation
M = calibrate_and_estimate_M(dataset_params, models, val_grid);

%Apply exemplars on test set
test_files = apply_all_exemplars(dataset_params, models, test_set, ...
                                 testset_name, [], test_params, ...
                                 test_gt_function);

%Load testset results
test_grid = load_result_grid(dataset_params, models, testset_name, test_files);

%Evaluation of RAW SVM classifiers + DISPLAY
teststruct = pool_results(dataset_params,models,test_grid);
% [results] = evaluate_pascal_voc_grid(dataset_params, ...
%                                       models, test_grid, ...
%                                       testset_name, teststruct);
% show_top_dets(dataset_params, models, test_grid,...
%               test_set, testset_name, ...
%               teststruct);

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

if strcmp(VOCYEAR,'VOC2007')
  %% no evaluation for VOC2010 dataset.. need to do it online
  [results] = evaluate_pascal_voc_grid(dataset_params, ...
                                       models, test_grid, ...
                                       testset_name, teststruct);
else
  fprintf(1,'Evaluation on performed on VOC2007, not on %s\n', ...
          VOCYEAR);
end

%Show top detections
show_top_dets(dataset_params, models, test_grid,...
              test_set, testset_name, ...
              teststruct);
