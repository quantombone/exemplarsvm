function tg(models)
if ~exist('models','var')
  models=load_all_models_imex('','exemplars2','100');
  %models = load_all_models_chunks('train','exemplars2','100');
end

r = randperm(length(models));
%r = 9;
%r = 3;
% person with arms out
%r = 18;

%person with torso
r = 104;
for q = 1:length(r)
  train_regressor(models{r(q)});
end