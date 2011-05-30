function models = strip_model(models)
for i = 1:length(models)
  models{i}.model.x = [];
  models{i}.model.target_id = models{i}.model.target_id(1);
  models{i}.model.target_x = models{i}.model.target_x(:,1);

  %models{i}.model.svscores = models{i}.model.w(:)'*models{i}.model.nsv ...
  %    - models{i}.model.b;
  models{i}.model.nsv = [];
  models{i}.model.svids = [];
  models{i}.model.wtrace = [];
  models{i}.model.btrace = [];
end
