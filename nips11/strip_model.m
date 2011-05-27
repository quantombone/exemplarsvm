function models = strip_model(models)
for i = 1:length(models)
  models{i}.model.x = [];
  models{i}.model.target_id = [];
  models{i}.model.target_x = [];
  models{i}.model.nsv = [];
  models{i}.model.svids = [];
  models{i}.model.wtrace = [];
  models{i}.model.btrace = [];
end
