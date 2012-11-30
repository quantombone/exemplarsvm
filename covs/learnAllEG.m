function model = learnAllEG(data_set,cls,covstruct,add_flips,hg_size,A)
%Learn EvG models for objects of class cls inside the data_set
%covstruct is the covariance data structure and hg_size is the target
%hg_size if specified
%if cls == '', then use all classes

if ~exist('add_flips','var')
  add_flips = 1;
end

if exist('hg_size','var')

  subinds = get_subinds(covstruct,hg_size);
  lambda = .01;
  if ~exist('A') || numel(A) == 0
    A = inv(lambda*eye(length(subinds))+...
            covstruct.c(subinds, subinds));
  end
else
  hg_size = [];
  A = [];
end

for i = 1:length(data_set)
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


    model{iii} = learnGaussianTriggs({s}, s.objects(1).class, covstruct, ...
                                     0,hg_size,A);
    
    model{iii}.models{1}.gtbb = s.objects(1).bbox(1:4);
    model{iii}.models{1}.gtbb(11) = i;
    model{iii}.models{1}.gtbb(12) = 0;

    model{iii}.params = model{iii}.params;
    model{iii}.params.detect_add_flip = add_flips;
    model{iii}.params.detect_max_windows_per_exemplar = 100;
    model{iii}.models{1}.params = model{iii}.params;
    model{iii}.models{1}.cls = s.objects(1).class;
    model{iii}.models{1}.bb(11) = i;
    model{iii}.models{1} = rmfield(model{iii}.models{1},'params');

    iii = iii+1;
  end
  allmodels{i} = model;
end


if sum(cellfun(@(x)length(x),allmodels))==0
  fprintf(1,'No good positives\n');
  model = [];
  return;
end

allmodels = cellfun2(@(x)x(:)',allmodels);
model=cat(2,allmodels{:});

clear allmodels


goods = find(cellfun(@(x)length(x)>0,model));
model = model(goods);


models = cellfun2(@(x)x.models{1},model);
model = rmfield(model{1},{'cls','model_name'});
model.models = models;
model.data_set = data_set;

for i = 1:length(model.models)
  model.models{i}.bb(:,6) = i;
  model.models{i}.gtbb(:,6) = i;
end

