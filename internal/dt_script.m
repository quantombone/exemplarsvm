
%ok
if ~exist('trainval','var')
  trainval=load('~/projects/pascal/VOC2007/trainval.mat');
  trainval = trainval.data_set;
end

%classes = {'chair','tvmonitor','cow','motorbike','bus','sofa'};
classes = {'bottle','tvmonitor','bicycle','cow','motorbike','bus','chair'};

for i = 1:length(classes)
  cls = classes{i};

%[positive_set, negative_set] = ...
%    split_sets(trainval, cls);
%data_set = cat(2,data_set,negative_set');
data_set = trainval;


params = esvm_get_default_params;
params.display = 0;
params.dump_images = 0;
params.detect_max_windows_per_exemplar = 200;
params.train_max_negatives_in_cache = 30000;
params.max_number_of_positives = 2000;
params.train_max_mined_images = 5000;
params.train_max_negative_images = 100000;
params.train_svm_c = .01;
params.train_max_windows_per_iteration = 3000;
params.train_positives_constant = 1;
params.mine_from_negatives = 1;
params.mine_from_positives = 1;
params.mine_skip_positive_objects_os = .1;
params.mine_from_positives_do_latent_update = 1;
params.train_max_scale = 1.0;
params.train_max_images_per_iteration = 100;
params.init_params.init_function=@ ...
    esvm_initialize_fixedframe_exemplar;
params.init_params.MAX_POS_CACHE = 30000;
params.latent_os_thresh = 0.4;
%params.init_params.K = 50;
params.init_params.ADD_LR = 1;
params.training_function = @ ...
    (m)esvm_update_svm(esvm_update_positives(esvm_update_svm(m,.5)));

%hardcode mask size
%params.init_params.hg_size = [10 10 31];

filer = sprintf('~/Desktop/inits/%s.mat',cls);
if fileexists(filer)
  load(filer);
else
  model = esvm_initialize_exemplars(data_set, cls, params);
  model = unify_model(model);
  model.params.display = 1;
  save(filer,'model');
end

model.params = params;
model.params.display = 1;

filer = sprintf('~/Desktop/DT/two.%s.mat',cls);
if fileexists(filer)
  load(filer);
else
  model = esvm_train(model);
  save(filer,'model');
end

end