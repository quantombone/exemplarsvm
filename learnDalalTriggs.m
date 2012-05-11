
%ok
if ~exist('trainval','var')
  trainval=load('~/projects/pascal/VOC2007/trainval.mat');
  trainval = trainval.data_set;
end

%classes = {'chair','tvmonitor','cow','motorbike','bus','sofa'};
classes = {'bicycle','tvmonitor','bottle','bus','chair','bus','cow','bottle','tvmonitor','bicycle','motorbike'};
for i = 1:length(classes)
  cls = classes{i};

%[positive_set, negative_set] = ...
%    split_sets(trainval, cls);
%data_set = cat(2,data_set,negative_set');
data_set = trainval;

params = esvm_get_default_params;
params.display = 0;
params.dump_images = 0;
params.detect_max_windows_per_exemplar = 100;
params.max_number_of_positives = 2000;

params.train_max_negatives_in_cache = 30000;
params.train_max_mined_images = 5000;
params.train_max_negative_images = 10000;
params.train_max_mine_iterations = 100;
%params.train_svm_c = .0001;
params.train_svm_c = .01;
params.train_max_windows_per_iteration = 5000;
params.train_positives_constant = 1;
params.train_max_scale = 1.0;
params.train_max_images_per_iteration = 200;
params.detect_pyramid_padding = 2;
params.detect_levels_per_octave = 10;

params.mine_from_negatives = 1;
params.mine_from_positives = 1;
params.mine_skip_positive_objects_os = .1;
params.mine_from_positives_do_latent_update = 0;

params.latent_os_thresh = 0.5;
params.init_params.MAX_POS_CACHE = 5000;
params.init_params.ADD_LR = 1;
params.init_params.init_function = ...
    @esvm_initialize_fixedframe_exemplar;
%params.training_function = @ ...
%    (m)esvm_update_svm(esvm_update_positives(esvm_update_svm(m,.8)));

params.training_function = @ ...
    (m)esvm_update_positives(esvm_update_svm(esvm_update_positives(esvm_update_svm(m,.8))));

%hardcode mask size
%params.init_params.hg_size = [10 10 31];

filer = sprintf('~/Desktop/inits/%s.mat',cls);
if fileexists(filer)
  fprintf(1,'loading %s\n',filer);
  load(filer);
else
  model = esvm_initialize_exemplars(data_set, cls, params);
  model.params.display = 1;
  model = unify_model(model);
  save(filer,'model');
end

params.display = 1;
model.params = params;

model.models{1}.params = params;
model.models{1}.params.train_svm_c = model.params.train_svm_c;
model.models{1}.params.train_newton_iter = 10;
model.params.train_newton_iter = 10;

filer = sprintf('~/Desktop/DT/%s.mat',cls);
if fileexists(filer)
  load(filer);
else
  model = esvm_train(model);
  save(filer,'model');
end

filer = sprintf('~/Desktop/ESVM/%s.mat',cls);
if fileexists(filer)
  load(filer)
else
  model = mixtures_real(model);
  save(filer,'model');
end

end