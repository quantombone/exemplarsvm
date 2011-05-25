function do_train_mr(models,res)
%% train all exemplars

if ~exist('models','var')
  models = load_all_models;
  res = load_result_grid(models);
end

VOCinit;

mode = [models{1}.models_name 'I'];

for i = 1:length(res)
  for j = 1:length(res{i}.objids)
    res{i}.objids{j}.curid = res{i}.index;
  end
end

%%% do all svms here
svs = cellfun2(@(x)x.svs,res);
svs = cat(2,svs{:});
objids = cellfun2(@(x)x.objids,res);
objids = cat(2,objids{:});

mp = get_default_mining_params;
mp.extract_negatives = 1;
bg = get_pascal_bg('trainval');

results_directory = ...
    sprintf('%s/%s/',VOCopts.localdir,mode);
if ~exist(results_directory,'dir')
  fprintf(1,'Making directory %s\n',results_directory);
  mkdir(results_directory);
end

myRandomize;
rrr = randperm(length(models));

bg = get_pascal_bg('trainval');

for ix = 1:length(models)
  i = rrr(ix);
  filer = sprintf('%s/%s.%d.%s.mat',results_directory,...
                  models{i}.curid,...
                  models{i}.objectid,...
                  models{i}.cls);
  
  filerlock = [filer '.lock'];
  if fileexists(filer) || (mymkdir_dist(filerlock)==0)
    continue
  end
  
  filerI = sprintf('%s/%s.%d.%s.png',results_directory,...
                  models{i}.curid,...
                  models{i}.objectid,...
                  models{i}.cls);
  

  m = models{i};
  m = add_new_detections(m,svs,objids);
  
  %curfeats = reshape(models{i}.model.x,models{i}.model.hg_size);
  %m.model.mask = sum(curfeats.^2,3)>0;
  [m] = do_svm(m,mp);
  
  %update the name
  m.models_name = mode;
  
  save(filer,'m');
  Isv = get_sv_stack(m,bg,10,10);
  imwrite(Isv,filerI);

  rmdir(filerlock);
  %Isv = get_sv_stack(m,bg,10);
  %imagesc(Isv)
  %drawnow
end
