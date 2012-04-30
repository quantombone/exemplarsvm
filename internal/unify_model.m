function model2 = unify_model(model)
%Convert separate "initialized" Exemplar-SVM models into one big
%fat model which will start a DT-training run

for i = 1:length(model.models)
  bbs{i} = model.models{i}.bb;
  bbs{i}(:,6) = i+(length(model.models))*bbs{i}(:,7);
end
x = cellfun2(@(x)x.x,model.models);
x = cat(2,x{:});
bb = cat(1,bbs{:});
model2 = model;
model2.models = model2.models(1);
model2.models{1} = rmfield(model2.models{1},'I');
model2.models{1} = rmfield(model2.models{1},'curid');
model2.models{1} = rmfield(model2.models{1},'objectid');
model2.models{1} = rmfield(model2.models{1},'gt_box');
model2.models{1} = rmfield(model2.models{1},'sizeI');
model2.models{1} = rmfield(model2.models{1},'name');

model2.models{1}.x = x;
model2.models{1}.bb = bb;
model2.models{1}.svxs = zeros(size(x,1),0);
model2.models{1}.svbbs = zeros(0,12);
model2.models{1}.resc = get_canonical_bb(model,model);
model2.models{1}.mask = logical(1+0*model2.models{1}.mask(:));

model2.models{1}.savex = x;
model2.models{1}.savebb = bb;
model2 = esvm_update_positives(model2,1);

hg_size = model.params.init_params.hg_size;
model.models{1}.center = [0 0 10*hg_size(1) 10*hg_size(2)];



