
%ok
if ~exist('trainval','var')
  trainval=load('~/projects/pascal/VOC2007/trainval.mat');
  trainval = trainval.data_set;
end

cls = 'bicycle';%bus';
%[positive_set, negative_set] = ...
%    split_sets(trainval, cls);
%data_set = cat(2,data_set,negative_set');
data_set = trainval;
%classes = {'tvmonitor','bicycle','boat','motorbike','bus'};

params = esvm_get_default_params;
params.display = 0;
params.dump_images = 0;
params.detect_max_windows_per_exemplar = 200;
params.train_max_negatives_in_cache = 20000;
params.max_number_of_positives = 2000;
params.train_max_mined_images = 10000;
params.train_svm_c = .01;
params.train_max_windows_per_iteration = 3000;
params.train_positives_constant = 1;
params.mine_from_negatives = 1;
params.mine_from_positives = 1;
params.mine_skip_positive_objects_os = .1;
params.mine_from_positives_do_latent_update = 0;
params.train_max_scale = 1.0;
params.init_params.hg_size = [10 10 31];
params.init_params.init_function=@ ...
    esvm_initialize_fixedframe_exemplar;
params.init_params.K = 50;
params.init_params.ADD_LR = 1;

filer = sprintf('~/Desktop/inits/%s.mat',cls);
if fileexists(filer)
  load(filer);
else
  model = esvm_initialize_exemplars(data_set, cls, params);
  model = unify_model(model);
  model.params.display = 1;
  save(filer,'model');
end

filer = sprintf('~/Desktop/DT/%s.mat',cls);
%if fileexists(filer)
%  load(filer);
%else
model = esvm_train(model);
save(filer,'model');
%end

