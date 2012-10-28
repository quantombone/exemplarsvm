function model = learnAllEG(data_set,cls,res,hg_size)
%Learn EvG models for objects of class cls inside the data_set
%res is the covariance data structure and hg_size is the target
%hg_size if specified
%if cls == '', then use all classes

if exist('hg_size','var')
  subinds = get_subinds(res,hg_size);
  lambda = .01;
  A = inv(lambda*eye(length(subinds))+...
        res.c(subinds, subinds));
else
  hg_size = [];
  A = [];
end

%iii = 1;
parfor i = 1:length(data_set)
  saves = data_set{i};

  
  if ~isfield(saves,'objects')
    continue
  end

  fprintf(1,'image %d / %d\n',i,length(data_set));

  iii = 1;
  model = cell(0,1);
  for j = 1:length(saves.objects)
    s = saves;
    
    if length(cls) > 0
      if ~sum(strcmp(cls,s.objects(j).class))
        continue
      end
    end
    s.objects = s.objects(j);
    if s.objects(1).difficult == 1 || s.objects(1).truncated==1
      continue
    end


    model{iii} = learnGaussianTriggs({s}, s.objects(1).class, res, ...
                                     0,hg_size,A);
    model{iii}.params = model{iii}.params;
    model{iii}.params.detect_add_flip = 1;
    model{iii}.params.detect_max_windows_per_exemplar = 10;
    model{iii}.models{1}.params = model{iii}.params;
    model{iii}.models{1}.cls = s.objects(1).class;
    model{iii}.models{1}.bb(11) = i;
    model{iii}.models{1} = rmfield(model{iii}.models{1},'params');

    iii = iii+1;
  end
  allmodels{i} = model;
end


model=cat(2,allmodels{:});
clear allmodels



goods = find(cellfun(@(x)length(x)>0,model));
model = model(goods);

if length(model) == 0
  fprintf(1,'No good positives\n');
  model = [];
  return;
end

models = cellfun2(@(x)x.models{1},model);
model = rmfield(model{1},{'cls','model_name'});
model.models = models;
model.data_set = data_set;


