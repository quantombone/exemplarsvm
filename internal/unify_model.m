function model2 = unify_model(model)
%Convert separate "initialized" Exemplar-SVM models into one big
%fat model which will start a DT-training run

for i = 1:length(model.models)
  bbs{i} = model.models{i}.bb;
  bbs{i}(:,6) = i+(length(model.models))*bbs{i}(:,7);
  gts{i} = [model.models{i}.gt_box 0 i 0 0 0 0 model.models{i}.curid ...
            0];
  curx{i} = model.models{i}.x(:,1);
  curbb{i}(:,5) = i;
  curbb{i} = bbs{i}(1,:);

end

x = cellfun2(@(x)x.x,model.models);
bb = cat(1,bbs{:});
x = cat(2,x{:});

model2 = model;
model2.models = model2.models(1);
model2.models{1}.gts = cat(1,gts{:});
model2.models{1} = rmfield(model2.models{1},'I');
model2.models{1} = rmfield(model2.models{1},'curid');
model2.models{1} = rmfield(model2.models{1},'objectid');
model2.models{1} = rmfield(model2.models{1},'gt_box');
model2.models{1} = rmfield(model2.models{1},'sizeI');
model2.models{1} = rmfield(model2.models{1},'name');


model2.models{1}.x = cat(2,curx{:});
model2.models{1}.bb = cat(1,curbb{:});

model2.models{1}.svxs = zeros(size(x,1),0);
model2.models{1}.svbbs = zeros(0,12);
model2.models{1}.resc = get_canonical_bb(model,model);
model2.models{1}.mask = logical(1+0*model2.models{1}.mask(:));

model2.models{1}.savex = x;
model2.models{1}.savebb = bb;

model2.models{1}.w = reshape(mean(model2.models{1}.x,2), ...
                             size(model2.models{1}.w));
model2.models{1}.w = model2.models{1}.w - mean(model2.models{1}.w(: ...
                                                  ));

model2.models{1}.b = -100;
model2.models{1}.savebb(:,5) = 1:size(model2.models{1}.savebb,1);

%model2 = esvm_update_positives(model2,1);

hg_size = model.params.init_params.hg_size;
model2.models{1}.center = [0 0 10*hg_size(1) 10*hg_size(2)];

[aa,bb] = ismember(model2.models{1}.bb(:,[1:4 7 11]), ...
                   model2.models{1}.savebb(:,[1:4 7 11]),'rows');

model2.models{1}.bb(:,5) = bb;
model2.models{1}.curc = model2.models{1}.resc(bb,:);
