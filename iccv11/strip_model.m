function models = strip_model(models)
for i = 1:length(models)
  models{i}.model.x = [];
  %models{i}.model.target_bb = models{i}.model.target_bb(1,:);
  %models{i}.model.target_x = models{i}.model.target_x(:,1);

  models{i}.model.svxs = [];
  models{i}.model.svbbs = [];
  
  models{i}.model.wtrace = [];
  models{i}.model.btrace = [];
  
  models{i}.mining_stats = [];
  models{i}.train_set = [];
end
