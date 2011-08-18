function [models,M] = load_voc_models(cls)

%% Initialize dataset
VOCYEAR = 'VOC2007';
suffix = load_data_directory;
dataset_params = get_voc_dataset(VOCYEAR,suffix);
dataset_params.params = get_default_mining_params;
dataset_params.params.FLIP_LR = 0;
dataset_params.params.lpo = 5;
dataset_params.params.MAXSCALE = .5;

if 1
  %Initialize framing function
  init_params.sbin = 8;
  init_params.goal_ncells = 100;
  init_params.MAXDIM = 12;
  init_params.init_function = @initialize_goalsize_model;
  init_params.init_string = 'g';
  init_params.init_type = sprintf('%s-%d-%d',...
                                  init_params.init_string,...
                                  init_params.goal_ncells,...
                                  init_params.MAXDIM);
else
  %Initialize exemplar framing function
  init_params.sbin = 8;
  init_params.hg_size = [8 8];
  init_params.init_function = @initialize_fixedframe_model;
  init_params.init_string = 'f';
  init_params.init_type = sprintf('%s-%d-%d', ...
                                  init_params.init_string, ...
                                  init_params.hg_size(1), ...
                                  init_params.hg_size(2));
end
dataset_params.init_params = init_params;

%Initialize exemplar stream
dataset_params.stream_set_name = 'trainval';
dataset_params.stream_max_ex = 5000;

dataset_params.must_have_seg = 0;
dataset_params.must_have_seg_string = '';
dataset_params.model_type = 'exemplar';

%Create mining/validation/testing params as defaults
%dataset_params.params = get_default_mining_params;
%dataset_params.params.nnmode = 'normalizedhog';

if 1
  %Choose the training function (do_svm, do_rank, ...)
  %Disable NMS in training params
  dataset_params.mining_params = dataset_params.params;
  dataset_params.mining_params.training_function = @do_svm;
  dataset_params.mining_params.NMS_OS = 1.0;
  dataset_params.mining_params.MAXSCALE = 0.5;
  dataset_params.mining_params.TOPK = 100;
  dataset_params.mining_params.set_name = 'train';
  %optional cap
  %dataset_params.mining_params.set_maxk = 0;
end

if 1
  dataset_params.val_params = dataset_params.params;
  dataset_params.val_params.NMS_OS = 0.5;
  dataset_params.val_params.set_name = 'trainval';
  %optional cap
  %dataset_params.val_params.set_maxk = 10000;
end

if 1
  dataset_params.test_params = dataset_params.params;
  dataset_params.test_params.NMS_OS = 0.5;
  dataset_params.test_params.set_name = 'test';
  %optional cap
  %dataset_params.test_params.set_maxk = 0;
end

%Choose a short string to indicate the type of training run we are doing
dataset_params.models_name = ...
    [init_params.init_type ...
     dataset_params.must_have_seg_string ...
     '.' ...
     dataset_params.model_type];

%classes = {cls};
%classes = {'motorbike','bicycle','person'};
classes = {'train','bus','car'};
%classes = {'chair','sofa','diningtable','tvmonitor'};
%classes = {'tvmonitor'};
for z = 1:length(classes)
  cls = classes{z};
  
  filer = sprintf('%s/models/%s-%s-svm-stripped.mat',...
                  dataset_params.localdir,cls,dataset_params.models_name);
  
  load(filer);
  
  
  %%update models to local directory
  for i = 1:length(models)
    models{i}.I = sprintf(dataset_params.imgpath, ...
                          models{i}.curid);
  end
  
  filer = sprintf('%s/betas/%s-%s-svm-betas.mat',...
                  dataset_params.localdir,cls,dataset_params.models_name);
  load(filer);
  M.betas = betas;
  [aa,bb] = sort(betas(:,1),'descend');
  NNN = length(betas);
  models = models(bb(1:NNN));
  M.betas = M.betas(bb(1:NNN),:);
  
  allmodels{z} = models;
  allbetas{z} = betas(bb(1:NNN),:);

end


models = cat(2,allmodels{:});
M.betas = cat(1,allbetas{:});

fg = get_pascal_set(dataset_params, 'test','+car');%, ['+' cls]);
dataset_params.display = 1;

apply_all_exemplars(dataset_params,models,fg,'test');%,M);

