function tg(models)
if ~exist('models','var')
  models = load_all_models_chunks('train','exemplars2','100');
end

r = randperm(length(models));
for q = 1:length(r)
  train_regressor(models{r(q)});
end